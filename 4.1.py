#!/usr/bin/env python3
# Minimal VAE for OASIS (PyTorch) + sampling + optional UMAP
# Formats: .npy  or .nii/.nii.gz
import os, glob, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

# ==== 可选：支持 NIfTI ====
try:
    import nibabel as nib
except ImportError:
    nib = None

# ---------------- Config ----------------
DATA_ROOT = os.environ.get("OASIS_PATH", "/home/groups/comp3710/")
OUT_DIR   = os.environ.get("OUT_DIR", "./out_vae")
IMG_SIZE  = 128
LATENT_D  = 16
BATCH     = 64
EPOCHS    = 10
LR        = 1e-3
SEED      = 42
NUM_WORK  = 4

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------- Dataset --------------
class Oasis2D(Dataset):
    def __init__(self, root, img_size=128):
        self.paths = sorted(
            glob.glob(os.path.join(root, "**/*.npy"), recursive=True) +
            glob.glob(os.path.join(root, "**/*.nii"), recursive=True) +
            glob.glob(os.path.join(root, "**/*.nii.gz"), recursive=True)
        )
        if not self.paths:
            raise RuntimeError(f"No .npy/.nii(.gz) found under {root}")
        self.img_size = img_size

    def _load_arr(self, p):
        if p.endswith(".npy"):
            arr = np.load(p)
        else:
            if nib is None:
                raise RuntimeError("Please `pip install nibabel` to read .nii/.gz files.")
            arr = nib.load(p).get_fdata()
        arr = arr.astype(np.float32)

        # 3D -> 2D: 取中间切片；2D 保持不变
        if arr.ndim == 3:
            arr = arr[:, :, arr.shape[2] // 2]
        if arr.max() > 1.0 or arr.min() < 0.0:
            m, M = arr.min(), arr.max()
            arr = (arr - m) / (M - m + 1e-8)
        return arr  # [H,W], [0,1]

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        x = self._load_arr(self.paths[idx])
        x = torch.from_numpy(x)[None, ...]  # [1,H,W]
        # 统一尺寸，用双线性插值
        x = F.interpolate(x[None], size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)[0]
        return x, 0

# -------------- VAE --------------
class Encoder(nn.Module):
    def __init__(self, zdim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True),  # 128->64
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True), # 64->32
            nn.Conv2d(64,128, 4, 2, 1), nn.ReLU(True), # 32->16
            nn.Conv2d(128,256,4, 2, 1), nn.ReLU(True)  # 16->8
        )
        self.fc_mu    = nn.Linear(256*8*8, zdim)
        self.fc_logv  = nn.Linear(256*8*8, zdim)

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logv(h)

class Decoder(nn.Module):
    def __init__(self, zdim=16):
        super().__init__()
        self.fc = nn.Linear(zdim, 256*8*8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(True), # 8->16
            nn.ConvTranspose2d(128,64, 4,2,1), nn.ReLU(True), # 16->32
            nn.ConvTranspose2d(64, 32, 4,2,1), nn.ReLU(True), # 32->64
            nn.ConvTranspose2d(32, 1,  4,2,1), nn.Sigmoid()   # 64->128
        )

    def forward(self, z):
        h = self.fc(z).view(-1,256,8,8)
        return self.deconv(h)

class VAE(nn.Module):
    def __init__(self, zdim=16):
        super().__init__()
        self.enc, self.dec = Encoder(zdim), Decoder(zdim)

    def reparam(self, mu, logv):
        std = torch.exp(0.5*logv)
        return mu + std*torch.randn_like(std)

    def forward(self, x):
        mu, logv = self.enc(x)
        z = self.reparam(mu, logv)
        xr = self.dec(z)
        return xr, mu, logv, z

def loss_fn(xr, x, mu, logv):
    rec = F.binary_cross_entropy(xr, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp()) / x.size(0)
    return rec + kld, rec, kld

# -------------- Train + Visualize --------------
def main(data_root):
    ds = Oasis2D(data_root, IMG_SIZE)
    n  = len(ds)
    tr = int(0.9*n)
    g  = torch.Generator().manual_seed(SEED)
    dtr, dva = torch.utils.data.random_split(ds, [tr, n-tr], generator=g)

    dl_tr = DataLoader(dtr, batch_size=BATCH, shuffle=True,  num_workers=NUM_WORK, pin_memory=True)
    dl_va = DataLoader(dva, batch_size=BATCH, shuffle=False, num_workers=NUM_WORK, pin_memory=True)

    vae = VAE(LATENT_D).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=LR)

    best = 1e9
    for e in range(1, EPOCHS+1):
        vae.train(); s = 0.0
        for x,_ in dl_tr:
            x = x.to(device, non_blocking=True)
            xr, mu, logv, _ = vae(x)
            L, rec, kld = loss_fn(xr, x, mu, logv)
            opt.zero_grad(); L.backward(); opt.step()
            s += L.item()*x.size(0)
        tr_loss = s / tr

        vae.eval(); s = 0.0
        with torch.no_grad():
            for x,_ in dl_va:
                x = x.to(device)
                xr, mu, logv, _ = vae(x)
                L, _, _ = loss_fn(xr, x, mu, logv)
                s += L.item()*x.size(0)
        va_loss = s / (n - tr)
        print(f"[{e}/{EPOCHS}] train={tr_loss:.4f} val={va_loss:.4f}")

        if va_loss < best:
            best = va_loss
            torch.save(vae.state_dict(), os.path.join(OUT_DIR, "vae_best.pt"))

    # ---- Recon Grid ----
    vae.load_state_dict(torch.load(os.path.join(OUT_DIR, "vae_best.pt"), map_location=device))
    vae.eval()
    x,_ = next(iter(dl_va))
    x = x.to(device)
    with torch.no_grad():
        xr, _, _, _ = vae(x)
    grid = torch.cat([x[:8], xr[:8]], dim=0).cpu()  # 原图+重建
    save_image(grid, os.path.join(OUT_DIR, "recon_grid.png"), nrow=8)

    # ---- Random Samples ----
    with torch.no_grad():
        z = torch.randn(64, LATENT_D, device=device)
        xs = vae.dec(z).cpu()
    save_image(xs, os.path.join(OUT_DIR, "samples_grid.png"), nrow=8)

    # ---- UMAP (optional) ----
    try:
        import umap, matplotlib.pyplot as plt
        Z = []
        with torch.no_grad():
            for i,(xb,_) in enumerate(dl_va):
                xb = xb.to(device)
                _, _, _, zb = vae(xb)
                Z.append(zb.cpu())
                if i >= 20: break
        Z = torch.cat(Z, 0).numpy()
        U = umap.UMAP(random_state=SEED).fit_transform(Z)
        plt.figure(figsize=(5,5)); plt.scatter(U[:,0], U[:,1], s=5, alpha=0.7)
        plt.title("VAE Latent Manifold (UMAP)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "umap_latent.png"), dpi=200)
        plt.close()
    except Exception as e:
        print("Skip UMAP:", e)

    print("Done. Outputs ->", OUT_DIR)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DATA_ROOT = sys.argv[1]
    main(DATA_ROOT)
