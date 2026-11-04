
import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    DCGAN Generator.

    Input:
        Noise vector of shape (batch_size, latent_dim)

    Output:
        Generated RGB images of shape (batch_size, 3, 64, 64)
    """

    def __init__(self, latent_dim=100, channels=3):
        super().__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            # 1 x 1 -> 4 x 4
            nn.ConvTranspose2d(
                latent_dim, 512, kernel_size=4,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 4 x 4 -> 8 x 8
            nn.ConvTranspose2d(
                512, 256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8 x 8 -> 16 x 16
            nn.ConvTranspose2d(
                256, 128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16 x 16 -> 32 x 32
            nn.ConvTranspose2d(
                128, 64, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 32 x 32 -> 64 x 64
            nn.ConvTranspose2d(
                64, channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.Tanh()
        )

    def forward(self, noise):
        noise = noise.view(noise.size(0), self.latent_dim, 1, 1)
        return self.model(noise)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator.

    Input:
        RGB images of shape (batch_size, 3, 64, 64)

    Output:
        Probability of each image being real.
    """

    def __init__(self, channels=3):
        super().__init__()

        self.model = nn.Sequential(
            # 64 x 64 -> 32 x 32
            nn.Conv2d(
                channels, 64, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 x 32 -> 16 x 16
            nn.Conv2d(
                64, 128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 x 16 -> 8 x 8
            nn.Conv2d(
                128, 256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 x 8 -> 4 x 4
            nn.Conv2d(
                256, 512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 4 x 4 -> 1 x 1
            nn.Conv2d(
                512, 1, kernel_size=4,
                stride=1, padding=0, bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, images):
        output = self.model(images)
        return output.view(images.size(0), 1)


def weights_init(model):
    """
    DCGAN weight initialization.
    """
    classname = model.__class__.__name__

    if "Conv" in classname:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

    elif "BatchNorm" in classname:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
