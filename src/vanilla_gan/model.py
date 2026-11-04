
import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Vanilla GAN Generator.

    Input:
        Noise vector of shape (batch_size, latent_dim)

    Output:
        Generated RGB images of shape
        (batch_size, channels, image_size, image_size)
    """

    def __init__(self, latent_dim=100, image_size=64, channels=3):
        super().__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels

        self.output_dim = channels * image_size * image_size

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, self.output_dim),
            nn.Tanh()
        )

    def forward(self, noise):
        generated = self.model(noise)

        generated = generated.view(
            noise.size(0),
            self.channels,
            self.image_size,
            self.image_size
        )

        return generated


class Discriminator(nn.Module):
    """
    Vanilla GAN Discriminator.

    Input:
        RGB images of shape
        (batch_size, channels, image_size, image_size)

    Output:
        Probability of each image being real.
    """

    def __init__(self, image_size=64, channels=3):
        super().__init__()

        self.image_size = image_size
        self.channels = channels

        self.input_dim = channels * image_size * image_size

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, images):
        images = images.view(images.size(0), -1)

        return self.model(images)


def weights_init(model):
    """
    Initialize Vanilla GAN weights.
    """
    classname = model.__class__.__name__

    if "Linear" in classname:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0.0)

    elif "BatchNorm1d" in classname:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0.0)
