import os, argparse, time, glob, random
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# -----------------------
#   数据集（调色板重映射 + 轻量增强）
# -----------------------
class OasisSlices(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=160, ignore_index=255, aug=False):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        if len(self.img_paths) != len(self.mask_paths):
            raise RuntimeError(f"images/masks count mismatch: {len(self.img_paths)} vs {len(self.mask_paths)}")
        self.img_size = int(img_size)
        self.ignore_index = int(ignore_index)
        self.aug = bool(aug)
        self.fixed_palette = {0: 0, 85: 1, 170: 2}

    def __len__(self): return len(self.img_paths)

    def _resize(self, img, mask):
        if self.img_size > 0:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        return img, mask

    def _augment(self, img, mask):
        if not self.aug: return img, mask
        if np.random.rand() < 0.5:
            img = ImageOps.mirror(img); mask = ImageOps.mirror(mask)
        if np.random.rand() < 0.1:
            img = ImageOps.flip(img); mask = ImageOps.flip(mask)
        if np.random.rand() < 0.2:
            deg = float(np.random.uniform(-10, 10))
            img = img.rotate(deg, resample=Image.BILINEAR)
            mask = mask.rotate(deg, resample=Image.NEAREST)
        return img, mask

    def _load_pair(self, ip, mp):
        img = Image.open(ip).convert("L")
        msk = Image.open(mp)
        img, msk = self._resize(img, msk)
        img, msk = self._augment(img, msk)
        return np.array(img, dtype=np.uint8, copy=True), np.array(msk, dtype=np.int64, copy=True)

    def _remap_mask(self, arr: np.ndarray) -> np.ndarray:
        uvals = np.unique(arr)
        if np.isin(uvals, [0, 85, 170, self.ignore_index]).all():
            out = np.full_like(arr, fill_value=self.ignore_index)
            for k, v in self.fixed_palette.items():
                out[arr == k] = v
            out[arr == self.ignore_index] = self.ignore_index
            return out
        valid = [int(v) for v in uvals if v != self.ignore_index]
        lut = {val: i for i, val in enumerate(sorted(valid))}
        out = np.full_like(arr, fill_value=self.ignore_index)
        for val, i in lut.items(): out[arr == val] = i
        return out

    def __getitem__(self, idx):
        x, y = self._load_pair(self.img_paths[idx], self.mask_paths[idx])
        y = self._remap_mask(y)
        x = torch.from_numpy(x).float().unsqueeze(0) / 255.0   # [1,H,W]
        y = torch.from_numpy(y).long()                         # [H,W]
        return x, y


# -----------------------
#   UNet
# -----------------------
class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=3, base=48):
        super().__init__()
        self.enc1 = ConvBNReLU(in_ch, base);     self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBNReLU(base, base*2);    self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBNReLU(base*2, base*4);  self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBNReLU(base*4, base*8);  self.pool4 = nn.MaxPool2d(2)
        self.bott = ConvBNReLU(base*8, base*16)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2); self.dec4 = ConvBNReLU(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8,  base*4, 2, stride=2); self.dec3 = ConvBNReLU(base*8,  base*4)
        self.up2 = nn.ConvTranspose2d(base*4,  base*2, 2, stride=2); self.dec2 = ConvBNReLU(base*4,  base*2)
        self.up1 = nn.ConvTranspose2d(base*2,  base,   2, stride=2); self.dec1 = ConvBNReLU(base*2,  base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bott(self.pool4(e4))
        d4 = self.up4(b); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.head(d1)


# -----------------------
#   损失 & 指标
# -----------------------
def dice_per_class(logits, target, ignore_index=255, eps=1e-6):
    N, C, H, W = logits.shape
    prob = torch.softmax(logits, dim=1)
    valid = (target != ignore_index)
    if not valid.any():
        return torch.zeros(C, device=logits.device, dtype=logits.dtype)
    t = target.clone(); t[~valid] = 0
    tgt = torch.nn.functional.one_hot(t, C).permute(0,3,1,2).float()
    valid = valid.unsqueeze(1).float()
    prob = prob * valid; tgt = tgt * valid
    dims = (0,2,3)
    inter = torch.sum(prob*tgt, dim=dims)
    denom = torch.sum(prob, dim=dims) + torch.sum(tgt, dim=dims)
    return (2*inter + eps)/(denom + eps)

class CELossDiceMix(nn.Module):
    def __init__(self, ce_w=0.5, dice_w=1.0, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ce_w, self.dice_w = float(ce_w), float(dice_w)
        self.ignore_index = int(ignore_index)
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        dsc = dice_per_class(logits, target, ignore_index=self.ignore_index).mean()
        return self.ce_w*ce + self.dice_w*(1.0 - dsc)


@torch.no_grad()
def evaluate(model, loader, device, save_dir=None, max_vis=6, ignore_index=255, palette=None):
    model.eval()
    total_c, count, saved = None, 0, 0
    if save_dir is not None:
        os.makedirs(os.path.join(save_dir, "viz"), exist_ok=True)
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x)
        dsc_c = dice_per_class(logits, y, ignore_index=ignore_index)
        if total_c is None: total_c = torch.zeros_like(dsc_c)
        total_c += dsc_c; count += 1
        if save_dir is not None and saved < max_vis:
            prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(1)  # [N,H,W]
            n = pred.size(0); i = 0
            while i < n and saved < max_vis:
                out_dir = os.path.join(save_dir, "viz"); os.makedirs(out_dir, exist_ok=True)
                base = f"sample_{saved:02d}"
                xi = (x[i,0].detach().float().cpu().clamp(0,1)*255).to(torch.uint8).numpy()
                yi = y[i].detach().cpu().numpy().astype(np.uint8)
                pi = pred[i].detach().cpu().numpy().astype(np.uint8)
                Image.fromarray(xi).save(os.path.join(out_dir, base+"_img.png"))
                Image.fromarray(yi).save(os.path.join(out_dir, base+"_gt.png"))
                Image.fromarray(pi).save(os.path.join(out_dir, base+"_pred.png"))
                C = int(logits.shape[1])
                oh = np.eye(C, dtype=np.uint8)[pi]; np.save(os.path.join(out_dir, base+"_pred_onehot.npy"), oh)
                palette_np = np.array([[0,0,0],[0,255,0],[255,0,0]], dtype=np.uint8) if palette is None else np.array(palette, dtype=np.uint8)
                color_pred = palette_np[pi]; rgb = np.stack([xi,xi,xi], axis=-1)
                blend = (0.6*rgb + 0.4*color_pred).astype(np.uint8)
                Image.fromarray(blend).save(os.path.join(out_dir, base+"_overlay.png"))
                saved += 1; i += 1
    mean_c = (total_c/max(count,1)).detach().cpu().numpy()
    return mean_c, float(mean_c.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='C:/Users/ALIENWARE/Desktop/demo2/OASIS')
    ap.add_argument('--save_dir', type=str, default='C:/Users/ALIENWARE/Desktop/demo2/runs_unet')
    ap.add_argument('--img_size', type=int, default=160)
    ap.add_argument('--num_classes', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--step_size', type=int, default=10)
    ap.add_argument('--gamma', type=float, default=0.1)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--channels_last', action='store_true')
    ap.add_argument('--val_interval', type=int, default=1)
    ap.add_argument('--ignore_index', type=int, default=255)
    ap.add_argument('--base', type=int, default=48)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    tr_img = os.path.join(args.data_root, "keras_png_slices_train")
    tr_msk = os.path.join(args.data_root, "keras_png_slices_seg_train")
    va_img = os.path.join(args.data_root, "keras_png_slices_validate")
    va_msk = os.path.join(args.data_root, "keras_png_slices_seg_validate")
    te_img = os.path.join(args.data_root, "keras_png_slices_test")
    te_msk = os.path.join(args.data_root, "keras_png_slices_seg_test")

    ds_tr = OasisSlices(tr_img, tr_msk, img_size=args.img_size, ignore_index=args.ignore_index, aug=True)
    ds_va = OasisSlices(va_img, va_msk, img_size=args.img_size, ignore_index=args.ignore_index, aug=False)
    ds_te = OasisSlices(te_img, te_msk, img_size=args.img_size, ignore_index=args.ignore_index, aug=False)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=32, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=32, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = UNet(in_ch=1, num_classes=args.num_classes, base=args.base).to(device)
    if args.channels_last: model = model.to(memory_format=torch.channels_last)

    criterion = CELossDiceMix(ce_w=0.5, dice_w=1.0, ignore_index=args.ignore_index)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.step_size, gamma=args.gamma)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    print(f"Device={device}, AMP={args.amp}, channels_last={args.channels_last}")
    print(f"Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}, Img={args.img_size}, C={args.num_classes}, base={args.base}")

    best_val, ep, t0 = -1.0, 1, time.time()
    while ep <= args.epochs:
        model.train()
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            if args.channels_last: x = x.to(memory_format=torch.channels_last)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                logits = model(x)
                if not hasattr(model, "_checked_labels"):
                    uvals = torch.unique(y).detach().cpu().tolist(); C = int(logits.shape[1])
                    print(f"[Label Check] unique labels in batch: {uvals}, num_classes={C}, ignore_index={args.ignore_index}")
                    bad = [v for v in uvals if (v != args.ignore_index and (v < 0 or v >= C))]
                    if bad: raise ValueError(f"Found out-of-range labels {bad}.")
                    model._checked_labels = True
                loss = criterion(logits, y)
            scaler.scale(loss).backward(); scaler.step(optim); scaler.update()

        if args.val_interval > 0 and (ep % args.val_interval == 0):
            m_c, m = evaluate(model, dl_va, device, save_dir=None, max_vis=0, ignore_index=args.ignore_index)
            print(f"[Val] Epoch {ep} | meanDSC={m:.4f} | per-class={np.round(m_c,4)}")
            print("[PASS] all classes > 0.9" if bool((m_c > 0.9).all()) else "[WARN] some classes < 0.9")
            if m > best_val:
                best_val = m
                torch.save({'state_dict': model.state_dict()}, os.path.join(args.save_dir, 'best.pth'))
                print("Saved best ->", os.path.join(args.save_dir, 'best.pth'))
        scheduler.step(); ep += 1

    ckpt_pth = os.path.join(args.save_dir, 'best.pth')
    if os.path.isfile(ckpt_pth):
        sd = torch.load(ckpt_pth, map_location='cpu')['state_dict']; model.load_state_dict(sd)
        print("Loaded best for test:", ckpt_pth)

    m_c, m = evaluate(model, dl_te, device, save_dir=args.save_dir, max_vis=9, ignore_index=args.ignore_index)
    ok_all = bool((m_c > 0.9).all()); dt = time.time() - t0
    print(f"[Test] meanDSC={m:.4f} | per-class={np.round(m_c,4)}")
    print("[PASS] all classes > 0.9" if ok_all else "[WARN] some classes < 0.9")
    print(f"Total Time: {dt/60:.1f} min")

    torch.save({'state_dict': model.state_dict()}, os.path.join(args.save_dir, 'final.pth'))
    print("Saved final ->", os.path.join(args.save_dir, 'final.pth'))


if __name__ == "__main__":
    main()
