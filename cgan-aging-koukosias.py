"""
CycleGAN Face Aging
====================
Dataset : UTKFace  (kaggle datasets download -d jangedoo/utkface-new)

Architecture
------------
  G   : Generator  young -> old   (ResNet encoder-9res-decoder)
  F   : Generator  old -> young   (same architecture, separate weights)
  D_Y : PatchGAN discriminator for the old domain
  D_X : PatchGAN discriminator for the young domain

Three losses keep training stable:
  1. Adversarial  (LSGAN / MSE)  — realism
  2. Cycle        (L1, w=10)     — identity preservation without paired data
  3. Identity     (L1, w=5)      — prevents unnecessary colour/background shifts

Usage
-----
  #Train
  python cyclegan_aging.py --mode train --data_dir ./UTKFace --epochs 50

  #Resume from checkpoint
  python cyclegan_aging.py --mode train --data_dir ./UTKFace \
                           --resume checkpoints/cyclegan_epoch025.pth

  #Run inference on your own photo
  python cyclegan_aging.py --mode infer \
                           --input my_photo.jpg \
                           --checkpoint checkpoints/cyclegan_epoch050.pth

  #Kaggle paths (dataset already mounted)
  python cyclegan_aging.py --mode train \
                           --data_dir /kaggle/input/utkface-new/UTKFace \
                           --out_dir  /kaggle/working/outputs \
                           --ckpt_dir /kaggle/working/checkpoints
"""
#--------------------------------------------------------------
#KOUKOSIAS ATHANASIOS 2025-2026 UTH
#FACE AGING CYCLE GAN-HEAVILY INFLUENCED BY 
#https://pub.towardsai.net/face-aging-using-conditional-gans-an-introduction-to-age-cgans-machine-learning-8a4a6a100201
#https://codinglabsong.medium.com/face-aging-cyclegan-from-paper-to-application-ba22269549de
#--------------------------------------------------------------

import os
import glob
import time
import random
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F_fn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


#--------------------------------------------------------------
#0. config (all hyperparameters) SOURCE:https://docs.python.org/3/library/argparse.html
#--------------------------------------------------------------

def get_config():
    parser = argparse.ArgumentParser(description="cgan-aging-koukosias")
    #mode
    parser.add_argument("--mode",choices=["train", "infer"], default="train")
    #paths
    parser.add_argument("--data_dir", default="./UTKFace",help="folder containing UTKFace .jpg files")
    parser.add_argument("--out_dir", default="./outputs",help="where to save sample grids and loss curves")
    parser.add_argument("--ckpt_dir", default="./checkpoints",help="where to save / load model checkpoints")
    parser.add_argument("--resume", default=None,help="path to checkpoint to resume training from")
    parser.add_argument("--checkpoint", default=None,help="checkpoint to use for --mode infer")
    parser.add_argument("--input", default=None,help="input image path for --mode infer")

    #architecture
    parser.add_argument("--img_size",type=int, default=128)
    parser.add_argument("--ngf",type=int, default=64,help="base filter count for generators")
    parser.add_argument("--ndf",type=int, default=64,help="base filter count for discriminators")
    parser.add_argument("--n_res",type=int, default=9,help="Residual blocks in generator (9 for 128px)")

    #training
    parser.add_argument("--epochs",type=int, default=50)
    parser.add_argument("--batch_size",type=int, default=4)
    parser.add_argument("--lr",type=float, default=2e-4)
    parser.add_argument("--lambda_cyc",type=float, default=10.0,help="cycle-consistency loss weight")
    parser.add_argument("--lambda_id",type=float, default=5.0,help="identity loss weight (paper recommends 0.5 * lambda_cyc)")
    parser.add_argument("--save_every",type=int, default=5,help="save checkpoint + sample grid every N epochs")
    parser.add_argument("--seed",type=int, default=42)

    #age domain boundaries
    parser.add_argument("--young_min",type=int, default=18)
    parser.add_argument("--young_max",type=int, default=28)
    parser.add_argument("--old_min",type=int, default=40)

    args = parser.parse_args()

    #cuda support if avail
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


#--------------------------------------------------------------
#1. dataset SOURCE:https://www.kaggle.com/datasets/jangedoo/utkface-new
#--------------------------------------------------------------

class UnpairedAgeDataset(Dataset):
    """
    Returns (young_img, old_img) pairs drawn independently.
    The two images are NEVER the same person — that is the whole point of CycleGAN.

    UTKFace filename format: [age]_[gender]_[race]_[timestamp].jpg
    Age split (default):
        Domain X (young) : age 18-28
        Domain Y (old)   : age 40+
    The gap (29-39) is deliberately excluded so the domains are clearly distinct.
    """
    def __init__(self, root_dir, young_min, young_max, old_min,transform=None, split="train", seed=42):
        self.transform = transform
        young_files, old_files = [], []
        for f in Path(root_dir).glob("*.jpg"):
            try:
                age = int(f.name.split("_")[0])
            except ValueError:
                continue
            if young_min <= age <= young_max:
                young_files.append(f)
            elif age >= old_min:
                old_files.append(f)

        #80 / 10 / 10 split so val/test
        def split_files(files):
            rng = random.Random(seed)
            files = sorted(files)
            rng.shuffle(files)
            n = len(files)
            cuts = {"train": files[:int(0.8 * n)],"val": files[int(0.8 * n):int(0.9 * n)],"test": files[int(0.9 * n):]}
            return cuts[split]

        self.young = split_files(young_files)
        self.old   = split_files(old_files)

        #balance domains so neither list loops faster than the other
        limit = min(len(self.young), len(self.old))
        self.young = self.young[:limit]
        self.old   = self.old[:limit]

        print(f"[{split:5s}] young: {len(self.young):,}  |  old: {len(self.old):,}")

    def __len__(self):
        return len(self.young)

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        #old image is chosen randomly --> guarantees unpaired batches
        old_idx = random.randint(0, len(self.old) - 1)
        return self._load(self.young[idx]), self._load(self.old[old_idx])

#data augmentation
def make_transforms(img_size, augment=True):
    if augment:
        return T.Compose([
            T.Resize((img_size + 30, img_size + 30), antialias=True),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.05, contrast=0.05),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
    else:
        return T.Compose([
            T.Resize((img_size + 30, img_size + 30), antialias=True),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])


#--------------------------------------------------------------
#2. generator  (ResNet encoder --> residual blocks --> decoder)
#--------------------------------------------------------------

class ResidualBlock(nn.Module):
    #two 3x3 convs with a skip connection. Spatial size is preserved.
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    ResNet generator used for BOTH directions (G and F have separate weights).

    Structure:
      Initial 7×7 conv         — large receptive field, captures face structure
      2 × stride-2 downsampling — compresses to feature space
      n_res residual blocks     — where aging transformation is learned
      2 × stride-2 upsampling  — restores spatial resolution
      final 7×7 conv + Tanh    — output in [-1, 1]

    ReflectionPad avoids the border artifacts that zero-padding introduces.
    InstanceNorm (not BatchNorm) works better for per-image style transfer.
    """
    def __init__(self, ngf=64, n_res=9):
        super().__init__()
        layers = []

        #initial block
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        #downsampling: 3 -> ngf -> ngf*2 -> ngf*4,  spatial /4
        in_ch = ngf
        for _ in range(2):
            out_ch = in_ch * 2
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        #residual blocks (aging is learned here)
        for _ in range(n_res):
            layers.append(ResidualBlock(in_ch))

        #upsampling: ngf*4 -> ngf*2 -> ngf,  spatial *4
        for _ in range(2):
            out_ch = in_ch // 2
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch,kernel_size=3, stride=2,padding=1, output_padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        #output layer
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, 3, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


#--------------------------------------------------------------
#3. discriminator-PatchGAN
#--------------------------------------------------------------

class Discriminator(nn.Module):
    """
    patchGAN discriminator
    judges small overlapping patches as real/fake instead of the whole image
    this forces the generator to produce realistic local textures (wrinkles, pores, hair) rather than just a globally plausible image
    returns a single logit per image (global average pool over patch scores)
    """
    def __init__(self, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            #no instanceNorm on first layer (raw image input)
            nn.Conv2d(3, ndf,   kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf,   ndf*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            #last conv: stride=1 to keep spatial resolution for patching
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            #one real/fake score per spatial patch
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        patch_scores = self.model(x) #(B, 1, H', W')
        #average all patch scores into a single number per image
        #note: using torch.nn.functional explicitly to avoid name collision with the Generator model also named 'F' in this script
        pooled = torch.nn.functional.avg_pool2d(patch_scores, patch_scores.shape[2:])
        return pooled.view(x.size(0), -1) #(B, 1)


#--------------------------------------------------------------------
#4. loss functions
#--------------------------------------------------------------------

mse_loss = nn.MSELoss()
l1_loss  = nn.L1Loss()

def adversarial_loss_D(real_pred, fake_pred):
    """
    LSGAN discriminator loss (MSE instead of BCE).
    D wants: real images -> score 1,  fake images -> score 0.
    MSE avoids vanishing gradients when D becomes very confident.
    """
    loss_real = mse_loss(real_pred, torch.ones_like(real_pred))
    loss_fake = mse_loss(fake_pred, torch.zeros_like(fake_pred))
    return (loss_real + loss_fake) * 0.5

def adversarial_loss_G(fake_pred):
    #generator wants D to score its fakes as real (-> 1)
    return mse_loss(fake_pred, torch.ones_like(fake_pred))

def cycle_loss(real, reconstructed):
    """
    L1 distance between original and round-trip reconstruction
    young -> G -> fake_old -> F -> recon_young  
    old   -> F -> fake_young -> G -> recon_old  
    This is what preserves identity without paired data
    """
    return l1_loss(reconstructed, real)

def identity_loss(real, same):
    """
    L1 distance when an image passes through the 'wrong' generator
    G(old_face) (already old, G shouldn't change it)
    F(young_face) (already young, F shouldn't change it)
    Prevents colour shifts and preserves background
    """
    return l1_loss(same, real)


#--------------------------------------------------------------------
#5. weights init
#--------------------------------------------------------------------

def weights_init(m):
    #normal init with mean=0, std=0.02 for conv layers (standard for GANs)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


#--------------------------------------------------------------------
#6. one training step
#--------------------------------------------------------------------

def train_step(real_young, real_old, G, F, D_Y, D_X,
               opt_G, opt_DY, opt_DX, cfg):
    """
    one batch update for all four models
    step 1 — generate fakes in both directions + cycle reconstructions + identity
    step 2 — update D_Y  (discriminate real old  vs fake old)
    step 3 — update D_X  (discriminate real young vs fake young)
    step 4 — update G+F  (adversarial + cycle + identity losses)
    returns a dict of scalar loss values for logging
    """
    real_young = real_young.to(cfg.device)
    real_old   = real_old.to(cfg.device)

    #step 1: forward pass
    fake_old    = G(real_young)   #young -> (fake) old
    fake_young  = F(real_old)     #old   -> (fake) young

    recon_young = F(fake_old)     #fake old   -> reconstructed young  (forward cycle)
    recon_old   = G(fake_young)   #fake young -> reconstructed old    (backward cycle)

    same_old    = G(real_old)     #old  through young2old G  -> should stay unchanged
    same_young  = F(real_young)   #young through old2young F -> should stay unchanged

    #step 2: update D_Y (discriminate real old  vs fake old)
    opt_DY.zero_grad()
    loss_DY = adversarial_loss_D(
        real_pred = D_Y(real_old),
        fake_pred = D_Y(fake_old.detach())
    )
    loss_DY.backward()
    opt_DY.step()

    #step 3: update D_X (discriminate real young vs fake young)
    opt_DX.zero_grad()
    loss_DX = adversarial_loss_D(
        real_pred = D_X(real_young),
        fake_pred = D_X(fake_young.detach())
    )
    loss_DX.backward()
    opt_DX.step()

    #step 4: update G and F jointly
    opt_G.zero_grad()

    loss_adv_G = adversarial_loss_G(D_Y(fake_old))     #G fools D_Y
    loss_adv_F = adversarial_loss_G(D_X(fake_young))   #F fools D_X

    loss_cyc_y = cycle_loss(real_young, recon_young)    #forward  cycle
    loss_cyc_o = cycle_loss(real_old,   recon_old)      #backward cycle

    loss_id_G  = identity_loss(real_old,   same_old)    #G(old)
    loss_id_F  = identity_loss(real_young, same_young)  #F(young) 

    loss_G_total = (
          loss_adv_G + loss_adv_F
        + cfg.lambda_cyc * (loss_cyc_y + loss_cyc_o)
        + cfg.lambda_id  * (loss_id_G  + loss_id_F)
    )
    loss_G_total.backward()
    opt_G.step()

    return {
        "G"  : loss_G_total.item(),
        "D_Y": loss_DY.item(),
        "D_X": loss_DX.item(),
        "cyc": (loss_cyc_y + loss_cyc_o).item(),
        "id" : (loss_id_G  + loss_id_F).item(),
    }


#--------------------------------------------------------------------
#7. visualisations
#--------------------------------------------------------------------

def denorm(t):
    #convert a [-1,1] tensor to a [0,1] numpy hwc array for imwrite
    return np.clip(t.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5, 0, 1)


def save_sample_grid(epoch, G, F, fixed_young, fixed_old, out_dir):
    """
    generate and save a 6 row grid for 4 fixed validation faces:
      row 1: real young
      row 2: fake old       (G output)
      row 3: reconstructed young (cycle)
      row 4: real old
      row 5: fake young     (F output)
      row 6: reconstructed old   (cycle)
    """
    G.eval(); F.eval()
    with torch.no_grad():
        fake_old    = G(fixed_young)
        fake_young  = F(fixed_old)
        recon_young = F(fake_old)
        recon_old   = G(fake_young)

    rows = {
        "real young"          : fixed_young,
        "fake old (G)"        : fake_old,
        "reconstructed young" : recon_young,
        "real old"            : fixed_old,
        "fake young (F)"      : fake_young,
        "reconstructed old"   : recon_old,
    }

    n_cols = 4
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(n_cols * 3, len(rows) * 3))
    fig.suptitle(f"epoch {epoch}", fontsize=14, y=1.01)

    for row_idx, (label, imgs) in enumerate(rows.items()):
        for col in range(n_cols):
            axes[row_idx][col].imshow(denorm(imgs[col]))
            axes[row_idx][col].axis("off")
            if col == 0:
                axes[row_idx][col].set_title(label, fontsize=9, loc="left")

    plt.tight_layout()
    path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"  --> sample grid saved: {path}")
    G.train(); F.train()


def save_loss_curves(history, out_dir):
    epochs_x = range(1, len(history["G"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs_x, history["G"],   color="steelblue")
    axes[0].set_title("Generator loss");  axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_x, history["D_Y"], color="tomato",   label="D_Y (old)")
    axes[1].plot(epochs_x, history["D_X"], color="orange",   label="D_X (young)")
    axes[1].set_title("Discriminator losses"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs_x, history["cyc"], color="seagreen", label="Cycle")
    axes[2].plot(epochs_x, history["id"],  color="purple",   label="Identity")
    axes[2].set_title("cycle + identity"); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.suptitle("training history", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  --> loss curves saved: {path}")


#--------------------------------------------------------------------
#8. training
#--------------------------------------------------------------------

def train(cfg):
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_dir,  exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    print(f"\ndevice : {cfg.device}")
    if cfg.device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    #data
    train_ds = UnpairedAgeDataset(
        cfg.data_dir, cfg.young_min, cfg.young_max, cfg.old_min,
        transform=make_transforms(cfg.img_size, augment=True),
        split="train", seed=cfg.seed
    )
    val_ds = UnpairedAgeDataset(
        cfg.data_dir, cfg.young_min, cfg.young_max, cfg.old_min,
        transform=make_transforms(cfg.img_size, augment=False),
        split="val", seed=cfg.seed
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=2,
                              pin_memory=(cfg.device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=2)

    #models
    G   = Generator(cfg.ngf, cfg.n_res).to(cfg.device)
    F   = Generator(cfg.ngf, cfg.n_res).to(cfg.device)
    D_Y = Discriminator(cfg.ndf).to(cfg.device)
    D_X = Discriminator(cfg.ndf).to(cfg.device)

    for model in [G, F, D_Y, D_X]:
        model.apply(weights_init)

    g_params = sum(p.numel() for p in G.parameters()) / 1e6
    d_params = sum(p.numel() for p in D_Y.parameters()) / 1e6
    print(f"\ngenerator (×2)     : {g_params:.1f}M params each")
    print(f"discriminator (×2) : {d_params:.1f}M params each")

    #optimizers
    #G and F share one optimizer — they're updated together in train_step
    #beta1=0.5: lower than default 0.9 so Adam forgets stale gradients faster
    opt_G  = optim.Adam(list(G.parameters()) + list(F.parameters()),
                        lr=cfg.lr, betas=(0.5, 0.999))
    opt_DY = optim.Adam(D_Y.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    opt_DX = optim.Adam(D_X.parameters(), lr=cfg.lr, betas=(0.5, 0.999))

    #linear LR decay: constant for first half, then ramp down to 0
    def lr_lambda(epoch):
        decay_start = cfg.epochs // 2
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / (cfg.epochs - decay_start))

    sched_G  = optim.lr_scheduler.LambdaLR(opt_G,  lr_lambda)
    sched_DY = optim.lr_scheduler.LambdaLR(opt_DY, lr_lambda)
    sched_DX = optim.lr_scheduler.LambdaLR(opt_DX, lr_lambda)

    #resume
    start_epoch = 1
    if cfg.resume:
        state = torch.load(cfg.resume, map_location=cfg.device)
        G.load_state_dict(state["G"])
        F.load_state_dict(state["F"])
        D_Y.load_state_dict(state["D_Y"])
        D_X.load_state_dict(state["D_X"])
        start_epoch = state["epoch"] + 1
        print(f"\nResumed from epoch {state['epoch']}: {cfg.resume}")

    #fixed validation batch for consistent progress grids
    fixed_young, fixed_old = next(iter(val_loader))
    fixed_young = fixed_young[:4].to(cfg.device)
    fixed_old   = fixed_old[:4].to(cfg.device)

    #save epoch-0 grid (untrained baseline)
    save_sample_grid(0, G, F, fixed_young, fixed_old, cfg.out_dir)

    #training loop
    history   = {"G": [], "D_Y": [], "D_X": [], "cyc": [], "id": []}
    n_batches = len(train_loader)

    print(f"\nTraining: {cfg.epochs} epochs × {n_batches} batches/epoch")
    print("─" * 60)

    for epoch in range(start_epoch, cfg.epochs + 1):
        G.train(); F.train(); D_Y.train(); D_X.train()
        sums = {k: 0.0 for k in history}
        t0   = time.time()

        pbar = tqdm(train_loader, total=n_batches,
                    desc=f"Epoch {epoch:3d}/{cfg.epochs}",
                    unit="batch", leave=True)

        for real_young, real_old in pbar:
            losses = train_step(real_young, real_old,
                                G, F, D_Y, D_X,
                                opt_G, opt_DY, opt_DX, cfg)
            for k in sums:
                sums[k] += losses[k]
            pbar.set_postfix(
                G   = f"{sums['G']   / (pbar.n + 1):.3f}",
                DY  = f"{sums['D_Y'] / (pbar.n + 1):.3f}",
                cyc = f"{sums['cyc'] / (pbar.n + 1):.3f}",
                refresh=False
            )

        pbar.close()
        sched_G.step(); sched_DY.step(); sched_DX.step()

        #epoch averages
        avgs = {k: sums[k] / n_batches for k in sums}
        for k in history:
            history[k].append(avgs[k])

        elapsed = time.time() - t0

        #training health check
        d_avg = (avgs["D_Y"] + avgs["D_X"]) / 2
        if d_avg < 0.05:
            status = "D too strong -- generators getting no learning signal"
        elif d_avg > 0.8:
            status = "G fooling D easily -- possible mode collapse"
        else:
            status = "balanced"

        print(
            f"  G={avgs['G']:.3f}  D_Y={avgs['D_Y']:.3f}  D_X={avgs['D_X']:.3f}"
            f"  cyc={avgs['cyc']:.3f}  id={avgs['id']:.3f}"
            f"  [{status}]  ({elapsed:.0f}s)"
        )

        #checkpoint + sample grid
        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            save_sample_grid(epoch, G, F, fixed_young, fixed_old, cfg.out_dir)
            ckpt_path = os.path.join(cfg.ckpt_dir, f"cyclegan_epoch{epoch:03d}.pth")
            torch.save({
                "G"    : G.state_dict(),
                "F"    : F.state_dict(),
                "D_Y"  : D_Y.state_dict(),
                "D_X"  : D_X.state_dict(),
                "epoch": epoch,
            }, ckpt_path)
            print(f"  --> checkpoint saved: {ckpt_path}")

    save_loss_curves(history, cfg.out_dir)
    print("\nTraining complete.")


#--------------------------------------------------------------------
#9. inference
#--------------------------------------------------------------------

def infer(cfg):
    if not cfg.input:
        raise ValueError("--input <path_to_image> is required for --mode infer")

    #find checkpoint
    ckpt_path = cfg.checkpoint
    if not ckpt_path:
        candidates = sorted(glob.glob(os.path.join(cfg.ckpt_dir, "cyclegan_epoch*.pth")))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in {cfg.ckpt_dir}")
        ckpt_path = candidates[-1]

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=cfg.device)

    G = Generator(cfg.ngf, cfg.n_res).to(cfg.device)
    F = Generator(cfg.ngf, cfg.n_res).to(cfg.device)
    G.load_state_dict(state["G"])
    F.load_state_dict(state["F"])
    G.eval(); F.eval()

    transform = make_transforms(cfg.img_size, augment=False)
    img   = Image.open(cfg.input).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(cfg.device)

    with torch.no_grad():
        aged   = G(img_t)   #young --> old
        younger = F(img_t)  #also produce younger version

    os.makedirs(cfg.out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(denorm(img_t.squeeze()));   axes[0].set_title("input");       axes[0].axis("off")
    axes[1].imshow(denorm(aged.squeeze()));    axes[1].set_title("aged (G)");    axes[1].axis("off")
    axes[2].imshow(denorm(younger.squeeze())); axes[2].set_title("younger (F)"); axes[2].axis("off")

    plt.suptitle("cycleGAN age progression", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(cfg.out_dir, "my_result.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"results saved --> {out_path}")


#--------------------------------------------------------------------
#10. entry point
#--------------------------------------------------------------------
def main():
    cfg = get_config()
    print(f"mode   : {cfg.mode}")
    print(f"device : {cfg.device}")
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "infer":
        infer(cfg)

if __name__ == "__main__":
    main()