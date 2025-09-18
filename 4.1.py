import os, time, math, argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader


# ---------------------------
#  Warmup + Cosine 调度器
# ---------------------------
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


# ---------------------------
#  数据集：OASIS Keras 切片
# ---------------------------
def _list_png(dir_path):
    names = []
    if os.path.isdir(dir_path):
        for n in os.listdir(dir_path):
            if n.endswith(".png"):
                names.append(n)
    return sorted(names)

def _pair_names(img_dir, msk_dir):
    imgs = _list_png(img_dir)
    msks = _list_png(msk_dir)
    a = set(imgs)
    b = set(msks)
    both = sorted(list(a.intersection(b)))
    if len(both) == 0:
        raise RuntimeError(f"[E] 未找到同名成对 PNG：\n  images={img_dir}\n  masks={msk_dir}")
    return both

class OASISKerasSlices(Dataset):
    """
    结构：
      root/
        keras_png_slices_train/         (灰度图像)
        keras_png_slices_seg_train/     (mask: 0/255 或 0/1)
        keras_png_slices_validate/
        keras_png_slices_seg_validate/
        keras_png_slices_test/
        keras_png_slices_seg_test/
    """
    def __init__(self, root, split="train", transform=None, num_classes=2):
        self.root = root
        self.split = split
        self.transform = transform
        self.num_classes = num_classes

        if split == "train":
            img_dir = os.path.join(root, "keras_png_slices_train")
            msk_dir = os.path.join(root, "keras_png_slices_seg_train")
        elif split == "val" or split == "validate":
            img_dir = os.path.join(root, "keras_png_slices_validate")
            msk_dir = os.path.join(root, "keras_png_slices_seg_validate")
        else:
            img_dir = os.path.join(root, "keras_png_slices_test")
            msk_dir = os.path.join(root, "keras_png_slices_seg_test")

        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.names = _pair_names(img_dir, msk_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        ip = os.path.join(self.img_dir, name)
        mp = os.path.join(self.msk_dir, name)

        img = np.array(Image.open(ip)).astype(np.float32)  # (H,W)
        msk = np.array(Image.open(mp)).astype(np.int64)    # (H,W)

        # Z-score
        m = float(img.mean()); s = float(img.std() + 1e-6)
        img = (img - m) / s

        # 二类分割：mask 归一化到 {0,1}
        if self.num_classes == 2:
            msk = (msk > 0).astype(np.int64)

        # To Tensor
        img = torch.from_numpy(img).unsqueeze(0)  # [1,H,W]
        msk = torch.from_numpy(msk)               # [H,W] long

        if self.transform:
            img = self.transform(img)
        return img, msk


# ---------------------------
#  简单的 Tensor 级增广/Resize
# ---------------------------
def make_tensor_augment(size):
    class TensorAug(object):
        def __init__(self, size): self.size = size
        def __call__(self, t):
            # t: [1,H,W]
            t = F.interpolate(t.unsqueeze(0), size=(self.size, self.size),
                              mode='bilinear', align_corners=False).squeeze(0)
            if torch.rand(1).item() < 0.5:
                t = torch.flip(t, dims=[2])  # 水平翻转
            return t
    return TensorAug(size)


# ---------------------------
#  UNet（简洁实现）
# ---------------------------
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


# ---------------------------
#  Dice + CE 复合损失 & 评估
# ---------------------------
class DiceCELoss(nn.Module):
    def __init__(self, num_classes, ce_weight=None, dice_weight=1.0, ce_weight_lambda=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.num_classes = num_classes
        self.dw = dice_weight
        self.cw = ce_weight_lambda

    def forward(self, logits, target):
        ce = self.ce(logits, target)  # CE
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target.clamp(min=0), self.num_classes).permute(0, 3, 1, 2).float()
        inter = (probs * onehot).sum(dim=(0, 2, 3)).sum()
        denom = (probs + onehot).sum(dim=(0, 2, 3)).sum()
        dice = 1.0 - (2 * inter + 1e-6) / (denom + 1e-6)
        return self.cw * ce + self.dw * dice

@torch.no_grad()
def dice_per_class(logits, target, num_classes, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target.clamp(min=0), num_classes).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    denom = (probs + onehot).sum(dim=(0, 2, 3))
    return (2 * inter + eps) / (denom + eps)  # [C]

@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    dices = torch.zeros(num_classes, device=device)
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        dices += dice_per_class(logits, y, num_classes)
        count += 1
    return (dices / max(count, 1)).mean().item()


# ---------------------------
#  可视化：叠图保存
# ---------------------------
def save_overlay(x, y_pred, y_true, out_png):
    # x:[1,H,W], y_pred/y_true:[H,W]
    x = x.squeeze(0).cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1); plt.imshow(x, cmap='gray'); plt.title('Image'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(y_true, vmin=0, vmax=y_pred.max()); plt.title('GT'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(y_pred, vmin=0, vmax=y_pred.max()); plt.title('Pred'); plt.axis('off')
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()


# ---------------------------
#  DataLoader 构造
# ---------------------------
def get_loaders(data_root, num_classes, batch_size, workers, use_channels_last, img_size=256):
    aug = make_tensor_augment(img_size)
    ds_tr = OASISKerasSlices(data_root, "train", transform=aug,                      num_classes=num_classes)
    ds_va = OASISKerasSlices(data_root, "val",   transform=make_tensor_augment(img_size), num_classes=num_classes)

    print(f"[Info] data_root={data_root}")
    print(f"[Info] train samples={len(ds_tr)}, val samples={len(ds_va)}")

    if len(ds_tr) == 0:
        raise RuntimeError("[E] 训练集为空，请检查数据路径与目录结构是否为 keras_png_slices_*")

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       num_workers=workers, pin_memory=True)
    return dl_tr, dl_va


# ---------------------------
#  主函数（while 驱动 epoch）
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True,
                    help='例如 /home/groups/comp3710/OASIS')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--num_classes', type=int, default=2)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--no_channels_last', action='store_true')
    ap.add_argument('--outdir', type=str, default='./runs_unet_keras')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    train_loader, val_loader = get_loaders(args.data_root, args.num_classes,
                                           args.batch_size, args.workers,
                                           not args.no_channels_last, img_size=args.img_size)

    model = UNet(in_ch=1, n_classes=args.num_classes).to(device)
    if not args.no_channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = DiceCELoss(args.num_classes, ce_weight=None, dice_weight=1.0, ce_weight_lambda=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.wd, nesterov=True)
    scheduler = WarmupCosine(optimizer, warmup_epochs=args.warmup, max_epochs=args.epochs)

    scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp)
    autocast = torch.amp.autocast('cuda', enabled=not args.no_amp)

    print(f"Device: {device}, AMP: {not args.no_amp}, channels_last: {not args.no_channels_last}")
    print(f"Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}, Warmup={args.warmup}")

    best_dice, ep = 0.0, 1
    t0 = time.time()

    while ep <= args.epochs:
        model.train()
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

        scheduler.step()

        val_dice = evaluate(model, val_loader, device, args.num_classes)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {ep}/{args.epochs} | LR: {cur_lr:.5f} | Val mDSC: {val_dice:.4f}")

        # 保存最好模型与少量可视化
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({'state_dict': model.state_dict(), 'dice': best_dice},
                       os.path.join(args.outdir, 'best_unet.pth'))
            # 保存 2 个样本的叠图
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
                                     os.path.join(args.outdir, f'viz_ep{ep}_{saved}.png'))
                        saved += 1
                        i += 1
                        if saved >= 2:
                            break
                    if saved >= 2:
                        break

        ep += 1

    print(f"Best Val mDSC: {best_dice:.4f}")
    print(f"Total Train Time: {time.time() - t0:.1f}s")

    torch.save({'state_dict': model.state_dict(), 'dice': best_dice},
               os.path.join(args.outdir, 'last_unet.pth'))


if __name__ == '__main__':
    main()
