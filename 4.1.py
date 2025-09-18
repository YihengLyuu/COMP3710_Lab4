# vae_oasis_min.py
import os, glob, argparse, numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

# ---------- 数据集（最小实现） ----------
IMG_EXTS = (".png", ".jpg", ".jpeg")
def list_files(root):
    fs = []
    for ext in IMG_EXTS + (".npy",):
        fs += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return sorted(fs)

class MRIDataset(Dataset):
    def __init__(self, root, size=128, max_files=None):
        self.size = size
        self.files = list_files(root)
        if max_files: self.files = self.files[:max_files]

    def __len__(self): return len(self.files)

    def _resize01(self, arr):
        arr = arr.astype(np.float32)
        if arr.max() > 0: arr = arr / arr.max()
        t = torch.from_numpy(arr)[None, None, ...]  # [1,1,H,W]
        t = F.interpolate(t, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return t[0]  # [1,S,S]

    def __getitem__(self, i):
        p = self.files[i].lower()
        if p.endswith(".npy"):
            arr = np.load(self.files[i], allow_pickle=False)
            arr = np.squeeze(arr)
            if arr.ndim == 3:  # (D,H,W) 取中间切片
                arr = arr[arr.shape[0] // 2]
            x = self._resize01(arr)
        else:
            img = Image.open(self.files[i]).convert("L").resize((self.size, self.size), Image.BILINEAR)
            x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)[None, ...]
        return x  # [1,S,S]

# ---------- 极简 VAE（2D 潜在） ----------
class VAE(nn.Module):
    def __init__(self, size=128, zdim=2):
        super().__init__()
        c = 32; s8 = size // 8
        self.enc = nn.Sequential(
            nn.Conv2d(1, c, 4, 2, 1), nn.ReLU(True),        # S/2
            nn.Conv2d(c, c*2, 4, 2, 1), nn.ReLU(True),      # S/4
            nn.Conv2d(c*2, c*4, 4, 2, 1), nn.ReLU(True)     # S/8
        )
        self.fc_mu  = nn.Linear(c*4*s8*s8, zdim)
        self.fc_log = nn.Linear(c*4*s8*s8, zdim)
        self.fc_dec = nn.Linear(zdim, c*4*s8*s8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c*4, c*2, 4, 2, 1), nn.ReLU(True),  # S/4
            nn.ConvTranspose2d(c*2, c,   4, 2, 1), nn.ReLU(True),  # S/2
            nn.ConvTranspose2d(c, 1,     4, 2, 1), nn.Sigmoid()    # S
        )
        self.size, self.zdim, self._c4, self._s8 = size, zdim, c*4, s8

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_log(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar); eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self._c4, self._s8, self._s8)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar, z

def kld(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# ---------- 可视化（重建 / 潜在散点 / 网格采样） ----------
@torch.no_grad()
def viz_recon(model, loader, out, tag="ep"):
    x = next(iter(loader))[:64].to(next(model.parameters()).device)
    xhat, *_ = model(x)
    save_image(make_grid(torch.cat([x, xhat], 0), nrow=8, pad_value=1.0), f"{out}/recon_{tag}.png")

@torch.no_grad()
def viz_scatter(model, loader, out):
    zs = []
    for x in loader:
        x = x.to(next(model.parameters()).device)
        mu, _ = model.encode(x)
        zs.append(mu.cpu().numpy())
        if sum(len(a) for a in zs) > 5000: break
    Z = np.concatenate(zs, 0)
    plt.figure(figsize=(5,5))
    plt.scatter(Z[:,0], Z[:,1], s=3, alpha=0.6)
    plt.title("Latent (mu)"); plt.xlabel("z1"); plt.ylabel("z2")
    plt.tight_layout(); plt.savefig(f"{out}/latent_scatter.png", dpi=200); plt.close()

@torch.no_grad()
def viz_grid(model, out, steps=20, lim=3.0):
    if model.zdim != 2: return
    xs = torch.linspace(-lim, lim, steps)
    z = torch.stack([torch.stack([x.repeat(steps), xs]) for x in xs], 0).reshape(2, -1).t().to(next(model.parameters()).device)
    xg = model.decode(z).cpu()
    save_image(make_grid(xg, nrow=steps, pad_value=1.0), f"{out}/latent_grid_{steps}x{steps}.png")

# ---------- 训练入口（最小参数集） ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="/home/groups/comp3710")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="./vae_out")
    ap.add_argument("--max_files", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = MRIDataset(args.data, size=args.size, max_files=args.max_files)
    n = len(full); n_tr = max(1, int(0.9*n)); n_va = n - n_tr
    tr, va = torch.utils.data.random_split(full, [n_tr, n_va], generator=torch.Generator().manual_seed(42))
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model = VAE(size=args.size, zdim=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs+1):
        model.train()
        loss_sum = 0.0
        for x in tl:
            x = x.to(device, non_blocking=True)
            xhat, mu, logvar, _ = model(x)
            recon = F.mse_loss(xhat, x)
            loss = recon + args.beta * kld(mu, logvar)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
        print(f"Epoch {ep}/{args.epochs} | loss={loss_sum/len(tr):.5f}")
        viz_recon(model, vl, args.out, tag=f"{ep:03d}")

    torch.save({"state_dict": model.state_dict(), "size": args.size}, f"{args.out}/vae.pth")
    viz_grid(model, args.out, steps=20, lim=3.0)
    viz_scatter(model, vl, args.out)
    print("Saved: vae.pth, recon_*.png, latent_grid_*.png, latent_scatter.png")

if __name__ == "__main__":
    main()
