import os
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from datetime import datetime
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, d=128, zdim=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.ConvTranspose2d(zdim, d*8, 4, 1, 0, bias=False),  # 1x1 -> 4x4
            nn.BatchNorm2d(d*8),
            nn.ReLU(True),
        )
        self.up1 = UpBlock(d*8, d*4)  # 4 -> 8
        self.up2 = UpBlock(d*4, d*2)  # 8 -> 16
        self.up3 = UpBlock(d*2, d)    # 16 -> 32
        self.up4 = UpBlock(d, d//2)   # 32 -> 64
        self.to_img = nn.Sequential(
            nn.Conv2d(d//2, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.up1(x); x = self.up2(x); x = self.up3(x); x = self.up4(x)
        return self.to_img(x)

G = Generator(128).to(device)

save_root = "/home/m/man0302/CS4243/project/CS4243-CAPTCHA/Captcha_DCGAN_results_20251111154440"
gen_weight_path = os.path.join(save_root, "generator_param.pkl")

G.load_state_dict(torch.load(gen_weight_path, map_location=device))
G.eval()


def tensor_to_pil(img_tensor):
    """
    img_tensor: shape (1, H, W) with values in [-1, 1]
    returns: PIL.Image in 'L' mode (grayscale)
    """
    img = img_tensor.squeeze(0).detach().cpu().numpy()  
    img = (img + 1.0) / 2.0 
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img, mode='L')
  
  
@torch.no_grad()
def generate_captcha(
    G,
    min_chars=5,
    max_chars=8,
    zdim=100,
    char_size=64,
    final_width=500,
    final_height=80
):
    """
    Generates a single captcha image:
      - sample N ~ Uniform[min_chars, max_chars]
      - generate N characters with G
      - concatenate horizontally
      - resize to (final_width, final_height)
    Returns:
      PIL.Image (L mode, size (final_width, final_height))
    """
    n_chars = random.randint(min_chars, max_chars)

    z = torch.randn(n_chars, zdim, 1, 1, device=device)
    fake_chars = G(z)                      # (n_chars, 1, 64, 64)

    pil_chars = [tensor_to_pil(fake_chars[i]) for i in range(n_chars)]

    captcha_width = n_chars * char_size
    captcha_height = char_size
    captcha_img = Image.new('L', (captcha_width, captcha_height), color=255)  

    x_offset = 0
    for im in pil_chars:
        if im.size != (char_size, char_size):
            im = im.resize((char_size, char_size), Image.BILINEAR)
        captcha_img.paste(im, (x_offset, 0))
        x_offset += char_size

    captcha_img = captcha_img.resize((final_width, final_height), Image.BILINEAR)

    return captcha_img
  
out_dir = "generated_captchas"
os.makedirs(out_dir, exist_ok=True)

for i in range(100):
    captcha = generate_captcha(G)
    ts = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
    captcha.save(os.path.join(out_dir, f"captcha_{ts}.png"))
    time.sleep(1)
    