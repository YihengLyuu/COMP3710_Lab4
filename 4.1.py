import os, time, math, argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader


# =========================
#  Warmup + Cosine 调度器
# =========================
class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch: int = -1):
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.max_epochs = int(max_epochs)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch + 1
        if e <= self.warmup_epochs and self.warmup_epochs > 0:
            return [base * e / self.warmup_epochs for base in self.base_lrs]
        if self.max_epochs <= self.warmup_epochs:
            return [base for base in self.base_lrs]
        t = (e - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
        t = min(max(t, 0.0), 1.0)
        return [base * 0.5 * (1.0 + math.cos(math.pi * t)) for base in self.base_lrs]


# =========================
#  UNet（简洁实现）
# =========================
class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=2, feat=(32, 64, 128, 256)):
        super().__init__()
        f1, f2, f3, f4 = feat
        self.enc1 = ConvBNReLU(in_ch, f1)
        self.enc2 = ConvBNReLU(f1, f2)
        self.enc3 = ConvBNReLU(f2, f3)
        self.enc4 = ConvBNReLU(f3, f4)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(f4, f3, 2, 2)
        self.dec3 = ConvBNReLU(f4, f3)
        self.up2 = nn.ConvTranspose2d(f3, f2, 2, 2)
        self.dec2 = ConvBNReLU(f3, f2)
        self.up1 = nn.ConvTranspose2d(f2, f1, 2, 2)
        self.dec1 = ConvBNReLU(f2, f1)
        self.head = nn.Conv2d(f1, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)  # [N,C,H,W] logits


# =========================
#  Dice + CE 复合损失（按类Dice，默认忽略背景）
# =========================
class DiceCELoss(nn.Module):
    def __init__(self, num_classes, ce_weight=None, dice_weight=1.0, ce_weight_lambda=1.0,
                 ignore_bg=True, eps=1e-6):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.num_classes = num_classes
        self.dw = dice_weight
        self.cw = ce_weight_lambda
        self.ignore_bg = ignore_bg
        self.eps = eps

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        probs = torch.softmax(logits, dim=1)                          # [N,C,H,W]
        onehot = F.one_hot(target.clamp(min=0), self.num_classes)     # [N,H,W,C]
        onehot = onehot.permute(0, 3, 1, 2).float()                   # [N,C,H,W]

        inter = (probs * onehot).sum(dim=(0, 2, 3))                   # [C]
        denom = (probs + onehot).sum(dim=(0, 2, 3))                   # [C]
        dice_c = (2 * inter + self.eps) / (denom + self.eps)          # [C]
        if self.ignore_bg and self.num_classes > 1:
            dice_c = dice_c[1:]
        dice = 1.0 - dice_c.mean()
        return self.cw * ce + self.dw * dice


@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_bg=True, eps=1e-6):
    """全验证集累计（macro Dice，可忽略背景）"""
    model.eval()
    inter_sum = torch.zeros(num_classes, device=device)
    denom_sum = torch.zeros(num_classes, device=device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)                          # [N,C,H,W]
        onehot = F.one_hot(y.clamp(min=0), num_classes).permute(0, 3, 1, 2).float()
        inter_sum += (probs * onehot).sum(dim=(0, 2, 3))
        denom_sum += (probs + onehot).sum(dim=(0, 2, 3))
    dice_c = (2 * inter_sum + eps) / (denom_sum + eps)
    if ignore_bg and num_classes > 1:
        dice_c = dice_c[1:]
    return dice_c.mean().item()


# =========================
#  可视化：叠图保存（vmax 对齐 GT/Pred）
# =========================
def save_overlay(x, y_pred, y_true, out_png):
    x = x.squeeze(0).cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    vmax = int(max(np.max(y_true), np.max(y_pred))) if (y_true.size and y_pred.size) else 1
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1); plt.imshow(x, cmap='gray'); plt.title('Image'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(y_true, vmin=0, vmax=vmax); plt.title('GT'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(y_pred, vmin=0, vmax=vmax); plt.title('Pred'); plt.axis('off')
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()


# ======================================================
#  数据加载（自动识别 Keras 切片 或 images/masks+txt）
# ======================================================
def _list_png(dir_path):
    names = []
    if os.path.isdir(dir_path):
        for n in os.listdir(dir_path):
            if n.endswith(".png"):
                names.append(n)
    return sorted(names)

def _pair_names(img_dir, msk_dir):
    a = set(_list_png(img_dir))
    b = set(_list_png(msk_dir))
    both = sorted(list(a.intersection(b)))
    if len(both) == 0:
        raise RuntimeError(f"[E] 未找到同名成对 PNG：\n  images={img_dir}\n  masks={msk_dir}")
    return both

class _OASIS_Keras_DS(Dataset):
    # 仅在检测到 keras_png_slices_* 结构时使用
    def __init__(self, img_dir, msk_dir, num_classes=2):
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.num_classes = num_classes
        self.names = _pair_names(img_dir, msk_dir)
    def __len__(self): return len(self.names)
    def __getitem__(self, idx):
        name = self.names[idx]
        ip = os.path.join(self.img_dir, name)
        mp = os.path.join(self.msk_dir, name)
        img = np.array(Image.open(ip)).astype(np.float32)   # (H,W)
        msk = np.array(Image.open(mp)).astype(np.int64)     # (H,W)
        # Z-score
        m, s = float(img.mean()), float(img.std() + 1e-6)
        img = (img - m) / s
        # mask 归一：二类→二值；多类→截断到合法范围
        if self.num_classes == 2:
            msk = (msk > 0).astype(np.int64)
        else:
            msk = np.clip(msk, 0, self.num_classes - 1).astype(np.int64)
        img = torch.from_numpy(img).unsqueeze(0)            # [1,H,W]
        msk = torch.from_numpy(msk)                         # [H,W]
        return img, msk

def _read_lines(p):
    out = []
    if os.path.isfile(p):
        with open(p, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    out.append(s)
    return out

class OASIS2DSeg(Dataset):
    """
    传统布局：
      data_root/
        images/*.png 或 .npy
        masks/*.png  或 .npy
        train.txt / val.txt  （文件名不含扩展名）
    """
    def __init__(self, root, ids, num_classes=2):
        self.root = root
        self.ids = ids
        self.num_classes = num_classes
        self.dir_images = os.path.join(root, "images")
        self.dir_masks = os.path.join(root, "masks")

    def _path(self, base, d):
        p_png = os.path.join(d, base + ".png")
        if os.path.isfile(p_png):
            return p_png, "png"
        p_npy = os.path.join(d, base + ".npy")
        if os.path.isfile(p_npy):
            return p_npy, "npy"
        raise FileNotFoundError(f"[E] 未找到样本：{base} 于 {d} (.png/.npy)")

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        bid = self.ids[idx]
        ip, itype = self._path(bid, self.dir_images)
        mp, mtype = self._path(bid, self.dir_masks)

        if itype == "png":
            img = np.array(Image.open(ip)).astype(np.float32)
        else:
            img = np.load(ip).astype(np.float32)

        if mtype == "png":
            msk = np.array(Image.open(mp)).astype(np.int64)
        else:
            msk = np.load(mp).astype(np.int64)

        m, s = float(img.mean()), float(img.std() + 1e-6)
        img = (img - m) / s

        if self.num_classes == 2:
            if msk.ndim > 2:
                msk = msk.squeeze()
            msk = (msk > 0).astype(np.int64)
        else:
            if msk.ndim > 2:
                msk = msk.squeeze()
            msk = np.clip(msk, 0, self.num_classes - 1).astype(np.int64)

        img = torch.from_numpy(img).unsqueeze(0)
        msk = torch.from_numpy(msk)
        return img, msk

def get_loaders(data_root, num_classes, batch_size, workers, use_channels_last):
    """
    自动识别两种布局：
    A) Keras 切片：
       data_root/
         keras_png_slices_train, keras_png_slices_seg_train
         keras_png_slices_validate, keras_png_slices_seg_validate
    B) 传统布局：
       data_root/
         images/, masks/, train.txt, val.txt
    """
    keras_train     = os.path.join(data_root, "keras_png_slices_train")
    keras_train_seg = os.path.join(data_root, "keras_png_slices_seg_train")
    keras_val       = os.path.join(data_root, "keras_png_slices_validate")
    keras_val_seg   = os.path.join(data_root, "keras_png_slices_seg_validate")

    if os.path.isdir(keras_train) and os.path.isdir(keras_train_seg) and \
       os.path.isdir(keras_val)   and os.path.isdir(keras_val_seg):
        ds_tr = _OASIS_Keras_DS(keras_train, keras_train_seg, num_classes=num_classes)
        ds_va = _OASIS_Keras_DS(keras_val,   keras_val_seg,   num_classes=num_classes)
        print(f"[Keras] train={len(ds_tr)}  val={len(ds_va)}  root={data_root}")
        if len(ds_tr) == 0:
            raise RuntimeError("[E] Keras 切片训练集为空，请检查目录与权限。")
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                           num_workers=workers, pin_memory=True, drop_last=True)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                           num_workers=workers, pin_memory=True)
        return dl_tr, dl_va

    images = os.path.join(data_root, "images")
    masks  = os.path.join(data_root, "masks")
    tr_txt = os.path.join(data_root, "train.txt")
    va_txt = os.path.join(data_root, "val.txt")
    if not (os.path.isdir(images) and os.path.isdir(masks)):
        raise FileNotFoundError(f"[E] 未检测到 Keras 结构，也未找到 images/ 与 masks/：{data_root}")

    ids_tr = _read_lines(tr_txt)
    ids_va = _read_lines(va_txt)
    if len(ids_tr) == 0 or len(ids_va) == 0:
        raise RuntimeError(f"[E] train.txt/val.txt 为空或缺失：\n{tr_txt}\n{va_txt}")

    ds_tr = OASIS2DSeg(data_root, ids_tr, num_classes=num_classes)
    ds_va = OASIS2DSeg(data_root, ids_va, num_classes=num_classes)
    print(f"[Legacy] train={len(ds_tr)}  val={len(ds_va)}  root={data_root}")

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       num_workers=workers, pin_memory=True)
    return dl_tr, dl_va


# =========================
#  主训练流程
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True, help='数据根目录（自动识别 Keras 或 Legacy 布局）')
    ap.add_argument('--num_classes', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--no_channels_last', action='store_true')
    ap.add_argument('--save_dir', type=str, default='./runs_unet_part4')
    args = ap.parse_args()

    # 复现性
    torch.manual_seed(42); np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    train_loader, val_loader = get_loaders(args.data_root, args.num_classes,
                                           args.batch_size, args.workers,
                                           not args.no_channels_last)

    model = UNet(in_ch=1, n_classes=args.num_classes).to(device)
    if not args.no_channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = DiceCELoss(args.num_classes, ce_weight=None, dice_weight=1.0,
                           ce_weight_lambda=1.0, ignore_bg=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=1e-4, nesterov=True)
    scheduler = WarmupCosine(optimizer, warmup_epochs=args.warmup, max_epochs=args.epochs)

    # 按你的要求：保持原 AMP 写法不改动
    scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp)
    autocast = torch.amp.autocast('cuda', enabled=not args.no_amp)

    print(f"Device: {device}, AMP: {not args.no_amp}, channels_last: {not args.no_channels_last}")
    print(f"Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}, Warmup={args.warmup}")

    best_dice, ep = 0.0, 1
    t0 = time.time()

    while ep <= args.epochs:
        model.train()
        loss_sum, n_batches = 0.0, 0

        for x, y in train_loader:
            if not args.no_channels_last:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast:
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.detach().float().item()
            n_batches += 1

        scheduler.step()

        val_dice = evaluate(model, val_loader, device, args.num_classes, ignore_bg=True)
        cur_lr = scheduler.get_last_lr()[0]
        tr_loss = loss_sum / max(n_batches, 1)
        print(f"Epoch {ep}/{args.epochs} | LR: {cur_lr:.5f} | Train Loss: {tr_loss:.4f} | Val mDSC: {val_dice:.4f}")

        # 保存最好模型与 2 张叠图
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'state_dict': model.state_dict(),
                'opt': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'dice': best_dice,
                'epoch': ep
            }, os.path.join(args.save_dir, 'best_unet.pth'))

            model.eval()
            saved = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    pred = model(x).argmax(1)
                    bsz = x.size(0)
                    i = 0
                    while i < bsz:
                        save_overlay(x[i].float().cpu(), pred[i].cpu(), y[i].cpu(),
                                     os.path.join(args.save_dir, f'viz_ep{ep}_{saved}.png'))
                        saved += 1
                        i += 1
                        if saved >= 2:
                            break
                    if saved >= 2:
                        break

        ep += 1

    print(f"Best Val mDSC: {best_dice:.4f}")
    print(f"Total Train Time: {time.time() - t0:.1f}s")

    torch.save({
        'state_dict': model.state_dict(),
        'opt': optimizer.state_dict(),
        'sched': scheduler.state_dict(),
        'dice': best_dice,
        'epoch': ep
    }, os.path.join(args.save_dir, 'last_unet.pth'))


if __name__ == '__main__':
    main()
