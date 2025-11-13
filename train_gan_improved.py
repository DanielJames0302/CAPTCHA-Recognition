import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.nn.utils import spectral_norm as SN
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SegmentedCharacterDataset(Dataset):
    """Custom dataset for loading segmented character images."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.png')]
        if len(self.image_files) == 0:
            raise ValueError(f"No PNG images found in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  
        if self.transform:
            image = self.transform(image)
        return image, 0  

img_size = 64
dataset_path = 'Datasets/full/train/segmented'
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))   # to [-1, 1]
])

custom_dataset = SegmentedCharacterDataset(root_dir=dataset_path, transform=transform)
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    custom_dataset, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=torch.cuda.is_available()
)

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

class Discriminator(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(
            SN(nn.Conv2d(1,   d,   4, 2, 1)),  nn.LeakyReLU(0.2, True),
            SN(nn.Conv2d(d,   d*2, 4, 2, 1)),  nn.LeakyReLU(0.2, True),
            SN(nn.Conv2d(d*2, d*4, 4, 2, 1)),  nn.LeakyReLU(0.2, True),
            SN(nn.Conv2d(d*4, d*8, 4, 2, 1)),  nn.LeakyReLU(0.2, True),
            SN(nn.Conv2d(d*8, 1,   4, 1, 0)), 
        )
    def forward(self, x):
        return self.net(x).view(-1)

def dcgan_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

G = Generator(128).to(device)
D = Discriminator(128).to(device)
G.apply(dcgan_init)
D.apply(dcgan_init)

def d_hinge_loss(d_real, d_fake):
    return F.relu(1. - d_real).mean() + F.relu(1. + d_fake).mean()

def g_hinge_loss(d_fake):
    return (-d_fake).mean()

G_optimizer = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
D_optimizer = optim.Adam(D.parameters(), lr=4e-4, betas=(0.0, 0.9))


fixed_z_ = torch.randn(25, 100, 1, 1, device=device)

@torch.no_grad()
def show_result(modelG, num_epoch, show=False, save=False, path='result.png', isFix=False):
    modelG.eval()
    z_ = fixed_z_ if isFix else torch.randn(25, 100, 1, 1, device=device)
    test_images = modelG(z_).cpu()
    modelG.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].axis("off")

    for k in range(25):
        i, j = divmod(k, 5)
        ax[i, j].imshow(test_images[k, 0].numpy(), cmap='gray', vmin=-1, vmax=1)

    label = f'Epoch {num_epoch}'
    fig.text(0.5, 0.04, label, ha='center')
    if save:
        plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))
    y1, y2 = hist['D_losses'], hist['G_losses']
    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')
    plt.xlabel('Iter'); plt.ylabel('Loss'); plt.legend(loc=4); plt.grid(True); plt.tight_layout()
    if save:
        plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


train_epoch = 200
ts = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
save_root = f'Captcha_DCGAN_results_{ts}'
os.makedirs(f'{save_root}/Random_results', exist_ok=True)
os.makedirs(f'{save_root}/Fixed_results', exist_ok=True)

train_hist = {'D_losses': [], 'G_losses': [], 'per_epoch_ptimes': [], 'total_ptime': []}
num_iter = 0

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses, G_losses = [], []
    epoch_start_time = time.time()

    for x_, _ in train_loader:
        x_ = x_.to(device)
        mini_batch = x_.size(0)

        # Train D
        D_optimizer.zero_grad(set_to_none=True)

        d_real = D(x_) 
        with torch.no_grad():
            z = torch.randn(mini_batch, 100, 1, 1, device=device)
            fake = G(z)
            
        d_fake = D(fake)  

        loss_d = d_hinge_loss(d_real, d_fake)
        loss_d.backward()
        D_optimizer.step()
        D_losses.append(loss_d.item())

        # Train G
        G_optimizer.zero_grad(set_to_none=True)

        z = torch.randn(mini_batch, 100, 1, 1, device=device)
        fake = G(z)
        d_fake_for_g = D(fake)

        loss_g = g_hinge_loss(d_fake_for_g)
        loss_g.backward()
        G_optimizer.step()
        G_losses.append(loss_g.item())

        num_iter += 1

    per_epoch_ptime = time.time() - epoch_start_time
    mean_D = float(torch.tensor(D_losses).mean())
    mean_G = float(torch.tensor(G_losses).mean())
    print(f'[{epoch+1}/{train_epoch}] - ptime: {per_epoch_ptime:.2f}, loss_d: {mean_D:.3f}, loss_g: {mean_G:.3f}')

    # save samples
    p = f'{save_root}/Random_results/Captcha_DCGAN_{epoch+1}.png'
    fixed_p = f'{save_root}/Fixed_results/Captcha_DCGAN_{epoch+1}.png'
    show_result(G, epoch+1, save=True, path=p, isFix=False)
    show_result(G, epoch+1, save=True, path=fixed_p, isFix=True)

    train_hist['D_losses'].append(mean_D)
    train_hist['G_losses'].append(mean_G)
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

total_ptime = time.time() - start_time
train_hist['total_ptime'].append(total_ptime)
print(f"Avg per epoch ptime: {float(torch.tensor(train_hist['per_epoch_ptimes']).mean()):.2f}, total {train_epoch} epochs ptime: {total_ptime:.2f}")
print("Training finish!... save training results")

# Save weights & curves
torch.save(G.state_dict(), f"{save_root}/generator_param.pkl")
torch.save(D.state_dict(), f"{save_root}/discriminator_param.pkl")
with open(f'{save_root}/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
show_train_hist(train_hist, save=True, path=f'{save_root}/Captcha_DCGAN_train_hist.png')

# ---- GIF ----
images = []
for e in range(train_epoch):
    img_name = f'{save_root}/Fixed_results/Captcha_DCGAN_{e+1}.png'
    images.append(imageio.v2.imread(img_name))
imageio.mimsave(f'{save_root}/generation_animation.gif', images, fps=5)
