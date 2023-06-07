import sys

import egg
import torch
import torchvision.transforms as transforms
from PIL import Image

interaction = torch.load(sys.argv[1])

transform = transforms.ToPILImage()

for index, sender_input in enumerate(interaction.sender_input):
    sender_input -= sender_input.min()
    sender_input /= sender_input.max()

    image: Image.Image = transform(sender_input)
    image.save(f"out/images/interaction_{index}.jpg", format="JPEG")
