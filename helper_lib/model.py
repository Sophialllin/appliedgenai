import torch.nn as nn
import torch
import math

# ---------------------------------------------------------------------------
# 1. CNN — (Assignment 2)
# ---------------------------------------------------------------------------
class AssignmentCNN(nn.Module):
    """CNN from Assignment 2.

    Input:   3 x 64 x 64
    Conv2d(3 -> 16, k=3, s=1, p=1) + ReLU   -> 16 x 64 x 64
    MaxPool2d(2, 2)                          -> 16 x 32 x 32
    Conv2d(16 -> 32, k=3, s=1, p=1) + ReLU  -> 32 x 32 x 32
    MaxPool2d(2, 2)                          -> 32 x 16 x 16
    Flatten                                  -> 8192
    Linear(8192 -> 100) + ReLU               -> 100
    Linear(100 -> 10)                        -> 10
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. GAN — (Assignment 3)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    GAN Generator.

    Architecture (from assignment3):
      Input : noise vector (BATCH_SIZE, 100)
      FC    : 100 → 7*7*128, reshape to (BATCH_SIZE, 128, 7, 7)
      ConvT : 128 → 64,  k=4, s=2, p=1  →  14×14  + BN + ReLU
      ConvT :  64 →  1,  k=4, s=2, p=1  →  28×28  + Tanh
    """

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.conv_blocks = nn.Sequential(
            # 128 → 64, 7×7 → 14×14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 → 1, 14×14 → 28×28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)               # (B, 7*7*128)
        x = x.view(-1, 128, 7, 7)   # (B, 128, 7, 7)
        return self.conv_blocks(x)   # (B, 1, 28, 28)


class Discriminator(nn.Module):
    """
    GAN Discriminator.

    Architecture (from assignment3):
      Input : image (1, 28, 28)
      Conv  :   1 →  64, k=4, s=2, p=1  →  14×14  + LeakyReLU(0.2)
      Conv  :  64 → 128, k=4, s=2, p=1  →   7×7   + BN + LeakyReLU(0.2)
      Flatten → Linear(128*7*7, 1)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 1 → 64, 28×28 → 14×14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 → 128, 14×14 → 7×7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # classify
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 1) — raw logit, no sigmoid


class GAN(nn.Module):
    """Container that holds both Generator and Discriminator."""

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

# ---------------------------------------------------------------------------
# 3. Diffusion Model — (Assignment 4)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    For timestep t and embedding dimension d, the i-th component is:
        emb[2i]   = sin(t / 10000^(2i/d))
        emb[2i+1] = cos(t / 10000^(2i/d))
    Then projected through a small MLP.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        ).float()
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


class ConvBlock(nn.Module):
    """Two Conv+GroupNorm+SiLU layers with time embedding injection and residual."""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.res_conv(x)


class UNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=64, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        ch = base_channels

        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        # Encoder
        self.enc_in = ConvBlock(img_channels, ch,   time_dim)   
        self.down1  = nn.Conv2d(ch,   ch*2, 3, stride=2, padding=1)
        self.enc1   = ConvBlock(ch*2, ch*2, time_dim)           
        self.down2  = nn.Conv2d(ch*2, ch*4, 3, stride=2, padding=1)
        self.enc2   = ConvBlock(ch*4, ch*4, time_dim)          
        self.down3  = nn.Conv2d(ch*4, ch*8, 3, stride=2, padding=1)
        self.enc3   = ConvBlock(ch*8, ch*8, time_dim)          

        # Bottleneck
        self.bottleneck = ConvBlock(ch*8, ch*8, time_dim)

        
        self.up3  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(ch*8 + ch*4, ch*4, time_dim)   
        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(ch*4 + ch*2, ch*2, time_dim)   
        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(ch*2 + ch,   ch,   time_dim)   

        self.out_conv = nn.Conv2d(ch, img_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)

        e0 = self.enc_in(x, t_emb)                
        e1 = self.enc1(self.down1(e0), t_emb)     
        e2 = self.enc2(self.down2(e1), t_emb)     
        e3 = self.enc3(self.down3(e2), t_emb)     

        h = self.bottleneck(e3, t_emb)            

        h = self.dec3(torch.cat([self.up3(h), e2], dim=1), t_emb)  
        h = self.dec2(torch.cat([self.up2(h), e1], dim=1), t_emb)  
        h = self.dec1(torch.cat([self.up1(h), e0], dim=1), t_emb)  
        return self.out_conv(h)

class DiffusionModel(nn.Module):
    """
    DDPM wrapper: UNet + fixed noise schedule (beta, alpha, alpha_bar buffers).
    """
    def __init__(self, img_channels=3, base_channels=64, time_dim=128,
                 T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T
        self.unet = UNet(img_channels, base_channels, time_dim)

        betas      = torch.linspace(beta_start, beta_end, T)
        alphas     = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas",      betas)
        self.register_buffer("alphas",     alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def forward(self, x, t):
        return self.unet(x, t)

    def q_sample(self, x0, t, noise=None):
        """Add noise to x0: x_t = sqrt(ā_t)*x0 + sqrt(1-ā_t)*noise"""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bars[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


# ---------------------------------------------------------------------------
# 4. Energy-Based Model (Assignment 4)
# ---------------------------------------------------------------------------

class EnergyModel(nn.Module):
    """
    CNN energy function E_theta(x) for CIFAR-10 (3x32x32 -> scalar).
    Lower energy = more data-like.
    """
    def __init__(self, img_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 32,  4, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,  64,  4, stride=2, padding=1),            # 8x8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  128, 4, stride=2, padding=1),            # 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=1, padding=0),            # 1x1
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)   # shape: (B,)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(model_name="CNN"):
    if model_name in ("CNN", "AssignmentCNN"):
        return AssignmentCNN()
    if model_name == "GAN":
        return GAN()
    if model_name == "Diffusion":
        return DiffusionModel()
    if model_name == "EBM":
        return EnergyModel()
    raise ValueError(
        f"Unknown model: {model_name!r}. "
        "Available: 'CNN', 'GAN', 'Diffusion', 'EBM'"
    )
