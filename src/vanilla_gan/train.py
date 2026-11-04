import torch

from vanilla_gan.model import Generator, Discriminator, weights_init


# Device configuration
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Hyperparameters
latent_dim = 100
batch_size = 16
image_size = 64
channels = 3


# Initialize Generator
generator = Generator(
    latent_dim=latent_dim,
    image_size=image_size,
    channels=channels
).to(device)


# Initialize Discriminator
discriminator = Discriminator(
    image_size=image_size,
    channels=channels
).to(device)


# Initialize model weights
generator.apply(weights_init)
discriminator.apply(weights_init)


# Generate random noise
noise = torch.randn(
    batch_size,
    latent_dim,
    device=device
)


# Generate fake images
generated_images = generator(noise)


# Print information
print("Device:", device)

print("Generator:", generator)

print("Discriminator:", discriminator)

print("Generated images shape:", generated_images.shape)
