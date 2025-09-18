import os, glob, time, argparse, math, numpy as np
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# -------------------------
# 一些小工具
# -------------------------
def read_any_image(fp):
    if fp.endswith(('.npy', '.npz')):
        arr = np.load(fp)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # 兼容 npz：优先取第一个数组
            arr = arr[list(arr.files)[0]]
        return arr
    else:
        return np.array(Image.open(fp))

def minmax01(x):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx > mn:
        x = (x - mn) / (mx - mn)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return x

# -------------------------
# 数据集
# -------------------------
class Oasis2DSeg(Dataset):
    def __init__(self, root, split='train', img_dir='images', mask_dir='masks',
                 img_size=256, val_ratio=0.1, seed=42):
        rng = np.random.RandomState(seed)
        img_glob = []
        for ext in ('*.npy','*.npz','*.png','*.jpg','*.jpeg','*.tif'):
            img_glob += glob.glob(os.path.join(root, img_dir, ext))
        img_glob.sort()
        assert len(img_glob) > 0, f"No images found under {os.path.join(root, img_dir)}"

        # 匹配同名 mask
        mask_list = []
        img_list  = []
        i = 0
        while i < len(img_glob):
            fp = img_glob[i]
            base = os.path.splitext(os.path.basename(fp))[0]
            found = None
            for ext in ('npy','npz','png','jpg','jpeg','tif'):
                cand = os.path.join(root, mask_dir, base + '.' + ext)
                if os.path.exists(cand):
                    found = cand; break
            if found is not None:
                img_list.append(fp); mask_list.append(found)
            i += 1

        n = len(img_list)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(1, int(n * val_ratio))
        val_idx = set(idx[:n_val])

        pairs = [(img_list[k], mask_list[k]) for k in range(n)]
        self.samples = []
        j = 0
        while j < len(pairs):
            if (split == 'train' and (j not in val_idx)) or (split == 'val' and (j in val_idx)) or (split=='test'):
                self.samples.append(pairs[j])
            j += 1

        self.img_size = int(img_size)
        self.to_tensor = transforms.ToTensor()  # [H,W] or [H,W,1] -> [1,H,W]

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        img_fp, msk_fp = self.samples[i]
        img = read_any_image(img_fp)
        msk = read_any_image(msk_fp)

        # 统一到 [H,W]
        if img.ndim == 3:  # 可能已有通道
            img = img[...,0]
        img = minmax01(img)
        msk = msk.astype(np.int64)

        # resize 到固定分辨率（双线性/最近邻）
        img_pil = Image.fromarray((img*255).astype(np.uint8))
        msk_pil = Image.fromarray(msk.astype(np.uint8))
        img_pil = img_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        msk_pil = msk_pil.resize((self.img_size, self.img_size), Image.NEAREST)

        img_t = self.to_tensor(img_pil)            # [1,H,W], float32 0..1
        msk_t = torch.from_numpy(np.array(msk_pil, dtype=np.int64))  # [H,W], long

        return img_t, msk_t

# -------------------------
# UNet (简洁 2D 版本)
# -------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=4, base_ch=32):
        super().__init__()
        self.enc1 = ConvBNReLU(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBNReLU(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBNReLU(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBNReLU(base_ch*4, base_ch*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBNReLU(base_ch*8, base_ch*16)

        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4 = ConvBNReLU(base_ch*16, base_ch*8)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = ConvBNReLU(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = ConvBNReLU(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = ConvBNReLU(base_ch*2, base_ch)

        self.head = nn.Conv2d(base_ch, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.head(d1)

# -------------------------
# Dice（one-hot）与组合损失
# -------------------------
def one_hot(labels, num_classes):
    # labels: [B,H,W] int64 -> [B,C,H,W] float
    b, h, w = labels.shape
    oh = torch.zeros((b, num_classes, h, w), dtype=torch.float32, device=labels.device)
    oh.scatter_(1, labels.unsqueeze(1), 1.0)
    return oh

def dice_per_class(probs, targets_oh, eps=1e-6):
    # probs: softmax [B,C,H,W], targets_oh: [B,C,H,W]
    dims = (0,2,3)
    inter = (probs * targets_oh).sum(dim=dims)
    union = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
    dice = (2.0*inter + eps) / (union + eps)
    return dice  # [C]

class CELossDice(nn.Module):
    def __init__(self, num_classes, ce_weight=None, dice_weight=1.0, ce_weight_factor=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.num_classes = num_classes
        self.dice_w = float(dice_weight)
        self.ce_w = float(ce_weight_factor)
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        probs = torch.softmax(logits, dim=1)
        target_oh = one_hot(target, self.num_classes)
        dpc = dice_per_class(probs, target_oh)
        dice_loss = 1.0 - dpc.mean()
        return self.ce_w*ce + self.dice_w*dice_loss, dpc.detach()

# -------------------------
# 评估：平均 Dice
# -------------------------
@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    tot = torch.zeros(num_classes, dtype=torch.float64, device=device)
    cnt = torch.zeros(num_classes, dtype=torch.float64, device=device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        d = dice_per_class(probs, one_hot(y, num_classes))
        tot += d.to(dtype=torch.float64)
        cnt += 1
    mean_dice = (tot / torch.clamp_min(cnt, 1)).mean().item()
    per_class = (tot / torch.clamp_min(cnt, 1)).tolist()
    return mean_dice, per_class

# -------------------------
# 可视化：叠加预测轮廓
# -------------------------
def save_preview(x, pred, gt, out_png):
    # x: [1,H,W], pred/gt: [H,W]
    x = (x.squeeze(0).cpu().numpy()*255).astype(np.uint8)
    pred = pred.cpu().numpy().astype(np.uint8)
    gt = gt.cpu().numpy().astype(np.uint8)
    # 用简单着色：边界上色
    from skimage.segmentation import find_boundaries
    from PIL import Image
    import numpy as np
    h, w = x.shape
    rgb = np.stack([x,x,x], axis=2)
    pb = find_boundaries(pred, mode='outer')
    gb = find_boundaries(gt, mode='outer')
    rgb[pb] = [255,0,0]   # 预测边界红
    rgb[gb] = [0,255,0]   # GT 边界绿
    Image.fromarray(rgb).save(out_png)

# -------------------------
# 训练主函数
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='/home/groups/comp3710/OASIS')
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--classes', type=int, default=4)  # 依据你的标签数量调整
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--no_channels_last', action='store_true')
    ap.add_argument('--predict_only', type=str, default='')
    ap.add_argument('--weights', type=str, default='unet_oasis.pth')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    ds_tr = Oasis2DSeg(args.data_root, 'train', img_size=args.img_size)
    ds_va = Oasis2DSeg(args.data_root, 'val',   img_size=args.img_size)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=8, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    model = UNet(1, args.classes, base_ch=32).to(device)
    if not args.no_channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.predict_only:
        ckpt = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        x, y = ds_va[0]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).argmax(1).squeeze(0)
        os.makedirs('viz', exist_ok=True)
        save_preview(x[0].detach().cpu(), pred.detach().cpu(), y, 'viz/predict_preview.png')
        print('Saved viz/predict_preview.png')
        return

    loss_fn = CELossDice(num_classes=args.classes, dice_weight=1.0, ce_weight_factor=1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp)
    autocast = torch.amp.autocast('cuda', enabled=not args.no_amp)

    print(f"Device={device}, AMP={not args.no_amp}, channels_last={not args.no_channels_last}")
    print(f"Train={len(ds_tr)} Val={len(ds_va)} Classes={args.classes}")

    best = 0.0
    ep = 1
    t0 = time.time()
    while ep <= args.epochs:
        model.train()
        it = iter(dl_tr)
        i = 0
        while i < len(dl_tr):
            x, y = next(it)
            if not args.no_channels_last:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast:
                logits = model(x)
                loss, _ = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            i += 1

        md, pcs = evaluate(model, dl_va, args.classes, device)
        print(f"Epoch {ep}/{args.epochs} | Val mDSC: {md:.4f} | per-class: {[round(v,4) for v in pcs]}")
        if md > best:
            torch.save({'state_dict': model.state_dict(), 'mDSC': md}, args.weights)
            best = md
        ep += 1

    dt = time.time() - t0
    print(f"Best Val mDSC: {best:.4f} | Total Train Time: {dt:.1f}s")
    print(f"Saved -> {args.weights}")

if __name__ == '__main__':
    main()
