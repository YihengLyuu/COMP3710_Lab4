import os, argparse, time, glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

# -----------------------
#   数据集
# -----------------------
class OasisSlices(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=160):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        assert len(self.img_paths) == len(self.mask_paths), "images/masks count mismatch"
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.img_paths)

    def _load_png(self, pth):
        # 读灰度
        img = Image.open(pth).convert("L")
        if self.img_size > 0:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        return arr

    def _load_mask(self, pth):
        msk = Image.open(pth)
        if self.img_size > 0:
            msk = msk.resize((self.img_size, self.img_size), Image.NEAREST)
        arr = np.asarray(msk, dtype=np.int64)  # 标签 id
        return arr

    def __getitem__(self, idx):
        x = self._load_png(self.img_paths[idx])
        y = self._load_mask(self.mask_paths[idx])
        # 归一化到[0,1]，再标准化（简单起见，仅/255）
        x = torch.from_numpy(x).float().unsqueeze(0) / 255.0  # [1,H,W]
        y = torch.from_numpy(y).long()                        # [H,W]
        return x, y

# -----------------------
#   UNet（简洁实现）
# -----------------------
class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=4, base=32):
        super().__init__()
        self.enc1 = ConvBNReLU(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBNReLU(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBNReLU(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBNReLU(base*4, base*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bott = ConvBNReLU(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = ConvBNReLU(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = ConvBNReLU(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBNReLU(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = ConvBNReLU(base*2, base)

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b  = self.bott(self.pool4(e4))

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

        logits = self.head(d1)  # [N,C,H,W]
        return logits

# -----------------------
#   Dice（逐类）与混合损失
# -----------------------
def one_hot(mask, num_classes):
    # mask:[N,H,W] -> [N,C,H,W]
    return torch.nn.functional.one_hot(mask, num_classes).permute(0,3,1,2).float()

def dice_per_class(logits, target, eps=1e-6):
    # logits: [N,C,H,W], target:[N,H,W]
    N, C, H, W = logits.shape
    prob = torch.softmax(logits, dim=1)
    tgt = one_hot(target, C)
    dims = (0,2,3)
    inter = torch.sum(prob * tgt, dim=dims)
    denom = torch.sum(prob, dim=dims) + torch.sum(tgt, dim=dims)
    dsc = (2*inter + eps) / (denom + eps)   # [C]
    return dsc

class CELossDiceMix(nn.Module):
    def __init__(self, ce_w=1.0, dice_w=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ce_w = ce_w
        self.dice_w = dice_w
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        dsc = dice_per_class(logits, target).mean()
        return self.ce_w*ce + self.dice_w*(1.0 - dsc)

# -----------------------
#   评估（每类 DSC）
# -----------------------
@torch.no_grad()
def evaluate(model, loader, device, save_dir=None, max_vis=6):
    model.eval()
    total_c = None
    count = 0
    saved = 0

    # 创建可视化目录
    if save_dir is not None:
        os.makedirs(os.path.join(save_dir, "viz"), exist_ok=True)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        dsc_c = dice_per_class(logits, y)   # [C]
        if total_c is None:
            total_c = torch.zeros_like(dsc_c)
        total_c += dsc_c
        count += 1

        # 可视化若干张
        if save_dir is not None and saved < max_vis:
            prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(1)  # [N,H,W]
            n = pred.size(0)
            i = 0
            while i < n and saved < max_vis:
                # 保存 3 张：原图、GT、Pred
                out_dir = os.path.join(save_dir, "viz")
                os.makedirs(out_dir, exist_ok=True)
                base = f"sample_{saved:02d}"
                # x[i] -> [1,H,W], 反归一化到[0,255]
                xi = (x[i,0].detach().float().cpu().clamp(0,1)*255).to(torch.uint8).numpy()
                yi = y[i].detach().cpu().numpy().astype(np.uint8)
                pi = pred[i].detach().cpu().numpy().astype(np.uint8)

                Image.fromarray(xi).save(os.path.join(out_dir, base+"_img.png"))
                Image.fromarray(yi).save(os.path.join(out_dir, base+"_gt.png"))
                Image.fromarray(pi).save(os.path.join(out_dir, base+"_pred.png"))
                saved += 1
                i += 1

    mean_c = (total_c / max(count,1)).detach().cpu().numpy()
    mean_dsc = float(mean_c.mean())
    return mean_c, mean_dsc

# -----------------------
#   主函数
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--save_dir', type=str, default='./runs_unet_oasis')
    ap.add_argument('--img_size', type=int, default=160)
    ap.add_argument('--num_classes', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--channels_last', action='store_true')
    ap.add_argument('--val_interval', type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    # 路径
    tr_img = os.path.join(args.data_root, "keras_png_slices_train")
    tr_msk = os.path.join(args.data_root, "keras_png_slices_seg_train")
    va_img = os.path.join(args.data_root, "keras_png_slices_validate")
    va_msk = os.path.join(args.data_root, "keras_png_slices_seg_validate")
    te_img = os.path.join(args.data_root, "keras_png_slices_test")
    te_msk = os.path.join(args.data_root, "keras_png_slices_seg_test")

    # DataLoader
    ds_tr = OasisSlices(tr_img, tr_msk, img_size=args.img_size)
    ds_va = OasisSlices(va_img, va_msk, img_size=args.img_size)
    ds_te = OasisSlices(te_img, te_msk, img_size=args.img_size)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=32, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=32, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # Model
    model = UNet(in_ch=1, num_classes=args.num_classes, base=32).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Loss & Optim
    criterion = CELossDiceMix(ce_w=1.0, dice_w=1.0)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    print(f"Device={device}, AMP={args.amp}, channels_last={args.channels_last}")
    print(f"Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}, Img={args.img_size}, C={args.num_classes}")

    best_val = -1.0
    ep = 1
    t0 = time.time()

    # 训练循环（不用 range）
    while ep <= args.epochs:
        model.train()
        for x, y in dl_tr:
            if args.channels_last:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        # 验证
        if args.val_interval > 0 and (ep % args.val_interval == 0):
            m_c, m = evaluate(model, dl_va, device, save_dir=None)
            print(f"[Val] Epoch {ep} | meanDSC={m:.4f} | per-class={np.round(m_c,4)}")
            if m > best_val:
                best_val = m
                torch.save({'state_dict': model.state_dict()}, os.path.join(args.save_dir, 'best.pth'))
                print("Saved best ->", os.path.join(args.save_dir, 'best.pth'))

        ep += 1

    # 最终测试 + 可视化
    ckpt_pth = os.path.join(args.save_dir, 'best.pth')
    if os.path.isfile(ckpt_pth):
        sd = torch.load(ckpt_pth, map_location='cpu')['state_dict']
        model.load_state_dict(sd)
        print("Loaded best for test:", ckpt_pth)

    m_c, m = evaluate(model, dl_te, device, save_dir=args.save_dir, max_vis=9)
    dt = time.time() - t0
    print(f"[Test] meanDSC={m:.4f} | per-class={np.round(m_c,4)}")
    print(f"Total Time: {dt/60:.1f} min")

    # 导出
    torch.save({'state_dict': model.state_dict()}, os.path.join(args.save_dir, 'final.pth'))
    print("Saved final ->", os.path.join(args.save_dir, 'final.pth'))

if __name__ == "__main__":
    main()
