import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model import Discriminator, Generator, initialize_weights
import tqdm

from PIL import Image
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 2e-4
batch_size = 64
image_size = 64
channels = 3
z_dim = 100
epochs = 50
features_disc = 64
features_gen = 64

transforms = T.Compose(
    [
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize([0.5 for _ in range(channels)], [0.5 for _ in range(channels)]),
    ]
)

dataset = ImageFolder(root='./dataset/', transform=transforms)
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

gen = Generator(z_dim, channels, features_gen).to(device)
disc = Discriminator(channels, features_disc).to(device)
initialize_weights(disc)
initialize_weights(gen)

optimizer_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_disc = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

def create_image_with_labels(fake_images, real_images):
    fake_images = fake_images.cpu().detach().numpy()
    real_images = real_images.cpu().detach().numpy()

    margin = 10
    image_size = 64

    image_width = 32*image_size+31*margin
    image_height = 4*image_size+3*margin
    background_color = (255, 255, 255)
    image = Image.new('RGB', (image_width, image_height), background_color)

    x = 0
    y = 0
    for i, img in enumerate(fake_images):
        img = img[0] * 255
        img = Image.fromarray(np.uint8(img))
        image.paste(img, (x, y))
        x += image_size + margin
        if i == 15 or i == len(fake_images)-1:
            x = 0
            y += image_size + margin

    for i, img in enumerate(real_images):
        img = img[0] * 255
        img = Image.fromarray(np.uint8(img))
        image.paste(img, (x, y))
        x += image_size + margin
        if i == 15 or i == len(real_images)-1:
            x = 0
            y += image_size + margin

    return image


for epoch in range(epochs):
    for (real, _) in tqdm.tqdm(loader):
        real = real.to(device)
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise)
        
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake)/2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        optimizer_disc.step()


        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        optimizer_gen.step()

    print(f'Epoch: {epoch}, Loss Disc: {lossD:.4f}, Loss Gen: {lossG:.4f}')

    with torch.no_grad():
        fake = gen(fixed_noise)

        image = create_image_with_labels(fake, real)
        image.save(f'./training_progress/epoch_{epoch}.png')
