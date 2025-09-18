import os, time, math, glob, numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils


# =============== 数据集：递归收集 .npy / .png / .jpg（更鲁棒） ===============
class OasisSlices(Dataset):
    def __init__(self, root, img_size=128):
        pats = ('*.npy', '*.png', '*.jpg', '*.jpeg')
        files = []
        i = 0
        while i < len(pats):
            files += glob.glob(os.path.join(root, '**', pats[i]), recursive=True)
            i += 1
        if len(files) == 0:
            raise FileNotFoundError(f'No images under: {root}')
        self.files = sorted(files)
        self.size = img_size
        self.to_tensor = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self): 
        return len(self.files)

    def _safe_load_npy(self, p):
        arr = np.load(p)  # 支持 HxW 或 CxHxW / HxWxC
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[0] not in (1, 3) and arr.shape[-1] in (1, 3):
            arr = np.moveaxis(arr, -1, 0)
        if arr.shape[0] != 1:
            arr = arr[:1]
        x = torch.from_numpy(arr).float()
        # 归一化到 [0,1]
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        # resize
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(self.size, self.size),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        return x

    def __getitem__(self, idx):
        p = self.files[idx]
        try:
            if p.lower().endswith('.npy'):
                return self._safe_load_npy(p)
            else:
                img = Image.open(p).convert('L')
                x = self.to_tensor(img)  # 1xHxW
                return x
        except (UnidentifiedImageError, OSError, ValueError) as e:
            # 遇到坏图/损坏文件：跳过 -> 返回一张全零图（不影响 batch 大小）
            print(f'[warn] skip unreadable file: {p} ({e})')
            return torch.zeros(1, self.size, self.size, dtype=torch.float32)


# ============================ VAE =============================
class VAE(nn.Module):
    def __init__(self, z=2):  # 固定 z=2，方便直接画潜在空间网格
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True),   # 64
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),  # 32
            nn.Conv2d(64, 128,4, 2, 1), nn.ReLU(True),  # 16
            nn.Conv2d(128,256,4, 2, 1), nn.ReLU(True),  # 8
        )
        self.flat = nn.Flatten()
        self.fc_mu  = nn.Linear(256*8*8, z)
        self.fc_lv  = nn.Linear(256*8*8, z)
        self.fc_up  = nn.Linear(z, 256*8*8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(True),   # 16
            nn.ConvTranspose2d(128,64,4,2,1),  nn.ReLU(True),   # 32
            nn.ConvTranspose2d(64,32,4,2,1),   nn.ReLU(True),   # 64
            nn.ConvTranspose2d(32,1,4,2,1),    nn.Sigmoid()     # 128
        )

    def encode(self, x):
        h = self.enc(x)
        h = self.flat(h)
        return self.fc_mu(h), self.fc_lv(h)

    def reparam(self, mu, lv):
        std = torch.exp(0.5*lv)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.fc_up(z).view(-1, 256, 8, 8)
        return self.dec(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        xrec = self.decode(z)
        return xrec, mu, lv


# ====================== 可视化（重建 & 流形） ======================
@torch.no_grad()
def save_recon(model, vis_batch, out_png):
    model.eval()
    x = vis_batch[:64]
    xrec, _, _ = model(x)
    grid = vutils.make_grid(torch.cat([x, xrec], 0), nrow=8, padding=2)
    vutils.save_image(grid, out_png)
    print(f'[viz] recon -> {out_png}')

@torch.no_grad()
def save_manifold(model, device, out_png, grid_n=10):
    # 仅 z=2 时生效：在 [-3,3]^2 均匀采样并解码
    xs = torch.linspace(-3, 3, steps=grid_n, device=device)
    ys = torch.linspace(-3, 3, steps=grid_n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    Z = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)  # (N,2)
    imgs, i, N, bs = [], 0, Z.size(0), 64
    autocast = torch.amp.autocast(device.type, enabled=(device.type == 'cuda'))
    while i < N:
        j = min(i+bs, N)
        with autocast:
            dec = model.decode(Z[i:j]).float().cpu()
        imgs.append(dec)
        i = j
    imgs = torch.cat(imgs, 0)
    grid = vutils.make_grid(imgs, nrow=grid_n, padding=2)
    vutils.save_image(grid, out_png)
    print(f'[viz] manifold -> {out_png}')


# ============================ 训练 =============================
def main():
    # 极简参数（其余都内置默认）
    data_dir = os.environ.get('OASIS_DIR', '/home/groups/comp3710/OASIS')  # 若课程给的共享盘目录不同，请改为实际路径
    epochs   = int(os.environ.get('EPOCHS', '40'))
    bsz      = int(os.environ.get('BATCH',  '128'))
    workers  = int(os.environ.get('WORKERS','4'))
    img_size = int(os.environ.get('SIZE',   '128'))
    beta     = float(os.environ.get('BETA', '1.0'))    # KL 权重（beta-VAE 可调）

    if not os.path.exists(data_dir):
        print(f'[error] data dir not found: {data_dir}')
        print('        请设置环境变量 OASIS_DIR 指向实际数据目录（本地或 Rangpur）。')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = (device.type == 'cuda')

    ds = OasisSlices(data_dir, img_size=img_size)
    loader = DataLoader(
        ds, batch_size=bsz, shuffle=True,
        num_workers=workers, pin_memory=(device.type == 'cuda'), drop_last=True
    )

    # 缓存一个稳定可视化 batch，避免每次重新取造成偶发问题
    vis_batch = None
    for xb in loader:
        vis_batch = xb
        break
    if vis_batch is None:
        print('[error] empty dataset after loading.')
        return

    model = VAE(z=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bce = nn.BCELoss(reduction='sum')

    scaler   = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))
    autocast = torch.amp.autocast(device.type, enabled=(device.type == 'cuda'))

    print(f'Device={device} | Samples={len(ds)} | Epochs={epochs} | Batch={bsz} | Size={img_size} | beta={beta}')

    t0, ep = time.time(), 1
    while ep <= epochs:
        model.train()
        tot_rec, tot_kl, tot_cnt = 0.0, 0.0, 0

        for x in loader:
            x = x.to(device, non_blocking=(device.type == 'cuda'))
            opt.zero_grad(set_to_none=True)
            with autocast:
                xrec, mu, lv = model(x)
                rec = bce(xrec, x)                                   # 重建
                kl  = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())# KL
                loss = rec + beta * kl
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tot_rec += rec.item(); tot_kl += kl.item(); tot_cnt += x.size(0)

        # 打印 epoch 级别指标（把 BCE 平均到每像素上以便直观）
        avg_rec = tot_rec / (tot_cnt * img_size * img_size)
        avg_kl  = tot_kl  /  tot_cnt
        print(f'Epoch {ep}/{epochs} | rec(BCE/pixel): {avg_rec:.5f} | KL/img: {avg_kl:.3f}')
        ep += 1

        # 可视化（用缓存的稳定 batch）
        try:
            save_recon(model, vis_batch.to(device), out_png=f'recon_ep{ep-1}.png')
            save_manifold(model, device, out_png=f'manifold_ep{ep-1}.png', grid_n=10)
        except Exception as e:
            print('[viz] skip:', e)

    dt = time.time() - t0
    torch.save({'state_dict': model.state_dict(), 'z_dim': 2, 'img_size': img_size}, 'vae_oasis_min.pth')
    print(f'Saved -> vae_oasis_min.pth | Total Train Time: {dt:.1f}s')


if __name__ == '__main__':
    main()
