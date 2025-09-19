import os, argparse, glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn


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
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1)); e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3)); b  = self.bott(self.pool4(e4))
        d4 = self.up4(b); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.head(d1)


def load_img_gray(pth, img_size=160):
    img = Image.open(pth).convert("L")
    if img_size > 0: img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    ten = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0) / 255.0  # [1,1,H,W]
    return arr, ten

@torch.no_grad()
def infer_one(model, device, in_pth, out_dir, img_size=160, palette=None):
    os.makedirs(out_dir, exist_ok=True)
    raw, ten = load_img_gray(in_pth, img_size=img_size)
    ten = ten.to(device)
    logits = model(ten)
    prob = torch.softmax(logits, dim=1)
    pred = prob.argmax(1)[0].detach().cpu().numpy().astype(np.uint8)

    Image.fromarray(raw).save(os.path.join(out_dir, "img.png"))
    Image.fromarray(pred).save(os.path.join(out_dir, "pred.png"))

    C = int(logits.shape[1])
    oh = np.eye(C, dtype=np.uint8)[pred]          # [H,W,C]
    np.save(os.path.join(out_dir, "pred_onehot.npy"), oh)

    palette_np = np.array([[0,0,0],[0,255,0],[255,0,0]], dtype=np.uint8) if palette is None else np.array(palette, dtype=np.uint8)
    color_pred = palette_np[pred]
    rgb = np.stack([raw,raw,raw], axis=-1)
    blend = (0.6*rgb + 0.4*color_pred).astype(np.uint8)
    Image.fromarray(blend).save(os.path.join(out_dir, "overlay.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', type=str, default='C:/Users/ALIENWARE/Desktop/demo2/runs_unet/best.pth')
    ap.add_argument('--input', type=str, default='C:/Users/ALIENWARE/Desktop/demo2/OASIS/keras_png_slices_test')
    ap.add_argument('--save_dir', type=str, default='C:/Users/ALIENWARE/Desktop/demo2/oasis_unet_out')
    ap.add_argument('--img_size', type=int, default=160)
    ap.add_argument('--num_classes', type=int, default=3)
    ap.add_argument('--base', type=int, default=48)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_ch=1, num_classes=args.num_classes, base=args.base).to(device)
    sd = torch.load(args.model_path, map_location='cpu')['state_dict']; model.load_state_dict(sd)
    model.eval()

    if os.path.isdir(args.input):
        paths = sorted(glob.glob(os.path.join(args.input, "*.png")))
        j = 0
        while j < len(paths):
            p = paths[j]
            name = os.path.splitext(os.path.basename(p))[0]
            out = os.path.join(args.save_dir, name)
            infer_one(model, device, p, out_dir=out, img_size=args.img_size)
            j += 1
    else:
        infer_one(model, device, args.input, out_dir=args.save_dir, img_size=args.img_size)

if __name__ == "__main__":
    main()
