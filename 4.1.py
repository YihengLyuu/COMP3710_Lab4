import os, sys, time, math, argparse, glob, csv
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.utils as vutils

# ---------------------------
# 实用：逐类 Dice（one-hot 语义分割），忽略背景可选
# ---------------------------
def dice_per_class(logits_softmax, target_onehot, eps=1e-8):
    # logits_softmax: [B,C,H,W], target_onehot: [B,C,H,W]
    B, C, H, W = logits_softmax.shape
    # 按通道做
    inter = (logits_softmax * target_onehot).sum(dim=(0,2,3))
    union = logits_softmax.sum(dim=(0,2,3)) + target_onehot.sum(dim=(0,2,3))
    dsc = (2.0 * inter + eps) / (union + eps)   # [C]
    return dsc

class DiceCELoss(nn.Module):
    def __init__(self, ce_weight=None, dice_bg_included=True, eps=1e-8):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_bg_included = dice_bg_included
        self.eps = eps
    def forward(self, logits, y): # logits [B,C,H,W], y [B,H,W] (long)
        ce = self.ce(logits, y)
        with torch.no_grad():
            C = logits.shape[1]
        prob = logits.softmax(1)
        onehot = torch.zeros_like(logits).scatter_(1, y.unsqueeze(1), 1.0)
        dsc_c = dice_per_class(prob, onehot, self.eps)  # [C]
        if not self.dice_bg_included and dsc_c.numel() > 1:
            dsc_used = dsc_c[1:]
        else:
            dsc_used = dsc_c
        dice_loss = 1.0 - dsc_used.mean()
        return ce + dice_loss, dsc_c

# ---------------------------
# 最小 2D UNet（下采样×4），基于 Conv3x3+BN+ReLU
# ---------------------------
def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, n_classes=4, base=32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(base*4, base*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bott = conv_block(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = conv_block(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = conv_block(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_block(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = conv_block(base*2, base)

        self.head = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bott(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

# ---------------------------
# OASIS 数据集（2D 切片版）
# 兼容三种常见预处理落地：*.npy|*.npz（H,W）/（C,H,W），或 *.png/*.jpg 成对（img, mask）
# 你只需把 data_root 指到包含 train/val 的目录
#   形如:
#   data_root/
#     imagesTr/*.npy|npz|png
#     labelsTr/*.npy|npz|png  (整型，0..C-1)
# ---------------------------
class Oasis2DSlices(Dataset):
    def __init__(self, img_paths, mask_paths, num_classes, to_channels_last=False):
        self.imgs = img_paths
        self.masks = mask_paths
        self.C = num_classes
        self.to_channels_last = to_channels_last

    def __len__(self):
        return len(self.imgs)

    def _load_arr(self, p):
        if p.endswith('.npy'):
            a = np.load(p)
        elif p.endswith('.npz'):
            a = np.load(p)['arr_0']
        else:
            # 读灰度
            a = np.array(Image.open(p).convert('L'))
        return a

    def __getitem__(self, idx):
        ip = self.imgs[idx]; mp = self.masks[idx]
        img = self._load_arr(ip)
        msk = self._load_arr(mp)

        # 统一到 [H,W] 灰度，若是 [C,H,W] 就取第一通道或做均值
        if img.ndim == 3:
            img = img.mean(axis=0)
        img = img.astype(np.float32)

        # 强度归一化到 0~1（鲁棒 min-max）
        vmin = np.percentile(img, 1.0); vmax = np.percentile(img, 99.0)
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        # mask 要求整型类别（0..C-1）
        if msk.ndim > 2:
            msk = msk.squeeze()
        msk = msk.astype(np.int64)
        msk = np.clip(msk, 0, self.C-1)

        # [1,H,W]
        img_t = torch.from_numpy(img).unsqueeze(0)
        msk_t = torch.from_numpy(msk)

        if self.to_channels_last:
            img_t = img_t.to(memory_format=torch.channels_last)

        return img_t, msk_t

def find_pairs(img_dir, mask_dir):
    # 用文件名（不含扩展名）对齐
    exts = ('.npy','.npz','.png','.jpg','.jpeg')
    imgs = []
    masks= []
    def stem(path): 
        b = os.path.basename(path)
        s = b[:b.rfind('.')]
        return s
    pool = {}
    for p in glob.glob(os.path.join(img_dir, '*')):
        if p.lower().endswith(exts):
            pool[stem(p)] = p
    keys = []
    for q in glob.glob(os.path.join(mask_dir, '*')):
        if q.lower().endswith(exts):
            k = stem(q)
            if k in pool:
                imgs.append(pool[k]); masks.append(q); keys.append(k)
    return imgs, masks

# ---------------------------
# Warmup + Cosine（与你 CIFAR 代码同逻辑）
# ---------------------------
class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
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

@torch.no_grad()
def evaluate(model, loader, device, num_classes, save_vis_dir=None, max_vis=4):
    model.eval()
    dices_sum = torch.zeros(num_classes, device=device)
    count = 0
    saved = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        prob = logits.softmax(1)
        onehot = torch.zeros_like(logits).scatter_(1, y.unsqueeze(1), 1.0)
        dsc_c = dice_per_class(prob, onehot)
        dices_sum += dsc_c
        count += 1

        if save_vis_dir is not None and saved < max_vis:
            pred = prob.argmax(1)  # [B,H,W]
            # 拼图：原图/GT/Pred
            grid = []
            bsz = x.shape[0]
            bi = 0
            while bi < bsz and saved < max_vis:
                # 归一化原图用于保存
                vis_img = x[bi, :1].detach()
                vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-6)
                # 把 GT / Pred 画成 one-hot 伪彩（取前三类叠 RGB）
                def to_rgb(label):
                    # 生成 [3,H,W]，把 1/2/3 映射到 R/G/B
                    H, W = label.shape
                    rgb = torch.zeros(3, H, W, device=label.device)
                    # 背景 0 忽略
                    rgb[0] = (label==1).float()
                    rgb[1] = (label==2).float()
                    rgb[2] = (label==3).float()
                    return rgb
                gt_rgb   = to_rgb(y[bi])
                pr_rgb   = to_rgb(pred[bi])
                # 3 张拼一起
                grid.append(torch.cat([vis_img.repeat(3,1,1), gt_rgb, pr_rgb], dim=2)) # [3,H,3W]
                bi += 1; saved += 1
            if grid:
                g = torch.cat(grid, dim=1) # [3, n*H, 3W]
                os.makedirs(save_vis_dir, exist_ok=True)
                vutils.save_image(g.cpu(), os.path.join(save_vis_dir, f'val_vis_{int(time.time())}.png'))
    mean_dsc = (dices_sum / max(count,1)).mean().item()
    return (dices_sum / max(count,1)).tolist(), mean_dsc

def get_loaders(data_root, num_classes, batch_size, workers, channels_last):
    train_img = os.path.join(data_root, 'imagesTr')
    train_lab = os.path.join(data_root, 'labelsTr')
    val_img   = os.path.join(data_root, 'imagesVal')
    val_lab   = os.path.join(data_root, 'labelsVal')
    Xtr, Ytr = find_pairs(train_img, train_lab)
    Xva, Yva = find_pairs(val_img,   val_lab)
    ds_tr = Oasis2DSlices(Xtr, Ytr, num_classes, to_channels_last=channels_last)
    ds_va = Oasis2DSlices(Xva, Yva, num_classes, to_channels_last=channels_last)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=8,        shuffle=False, num_workers=workers, pin_memory=True)
    return dl_tr, dl_va

def main():
    ap = argparse.ArgumentParser()
    # 你可按需要修改的关键超参
    ap.add_argument('--data_root', type=str, required=True, help='包含 imagesTr/labelsTr/imagesVal/labelsVal 的目录')
    ap.add_argument('--num_classes', type=int, default=4)               # 按数据集实际类别数设置（含背景）
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--no_channels_last', action='store_true')
    ap.add_argument('--save_dir', type=str, default='./oasis_unet_out')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    train_loader, val_loader = get_loaders(args.data_root, args.num_classes, args.batch_size, args.workers, not args.no_channels_last)

    model = UNet2D(in_ch=1, n_classes=args.num_classes, base=32).to(device)
    if not args.no_channels_last:
        model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = WarmupCosine(opt, warmup_epochs=args.warmup, max_epochs=args.epochs)
    criterion = DiceCELoss(ce_weight=None, dice_bg_included=True)

    scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp)
    autocast = torch.amp.autocast('cuda', enabled=not args.no_amp)

    print(f"Device: {device}, AMP: {not args.no_amp}, channels_last: {not args.no_channels_last}")
    print(f"Data: {args.data_root} | Classes={args.num_classes} | Epochs={args.epochs}")

    best_mean_dsc = -1.0
    ep = 1
    t0 = time.time()
    while ep <= args.epochs:
        model.train()
        for x, y in train_loader:
            if not args.no_channels_last:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast:
                logits = model(x)
                loss, dsc_c = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        # 验证 + 可视化
        dsc_c_list, mean_dsc = evaluate(model, val_loader, device, args.num_classes,
                                        save_vis_dir=os.path.join(args.save_dir, 'val_vis'), max_vis=2)
        cur_lr = opt.param_groups[0]['lr']
        print(f"Epoch {ep}/{args.epochs} | LR={cur_lr:.5e} | Val mDSC={mean_dsc:.4f} | per-class={['%.3f'%d for d in dsc_c_list]}")

        # 记录 CSV
        with open(os.path.join(args.save_dir, 'val_dsc.csv'), 'a', newline='') as f:
            w = csv.writer(f); w.writerow([ep, cur_lr, mean_dsc] + dsc_c_list)

        if mean_dsc > best_mean_dsc:
            best_mean_dsc = mean_dsc
            torch.save({'state_dict': model.state_dict(),
                        'num_classes': args.num_classes}, os.path.join(args.save_dir, 'best_dsc.pth'))
        ep += 1

    print(f"Best Val mDSC={best_mean_dsc:.4f}")
    print(f"Total Train Time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
