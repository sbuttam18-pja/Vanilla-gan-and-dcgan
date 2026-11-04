import torch

from dc_gan.model import Generator, Discriminator, weights_init


# Device configuration
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Hyperparameters
latent_dim = 100
batch_size = 64


# Initialize models
generator = Generator(
    latent_dim=latent_dim
).to(device)

discriminator = Discriminator().to(device)


# Initialize weights
generator.apply(weights_init)
discriminator.apply(weights_init)


# Test the generator
noise = torch.randn(
    batch_size,
    latent_dim,
    device=device
)

generated_images = generator(noise)

print("Device:", device)
print("Generated images:", generated_images.shape)
