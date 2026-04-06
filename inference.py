import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from model import Generator

# image paths
aliased_dir = 'datasets/OCMR/AL/R=8/test' # change before running
model_path = 'saved_models/generator-9.pth' # change before running

output_dir = 'generated_outputs/version-9' # change before running
os.makedirs(output_dir, exist_ok=True)

# load saved generator model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator().to(device)

generator.load_state_dict(torch.load(model_path))
generator.eval()

# define any transforms
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# test images
test_images = sorted([f for f in os.listdir(aliased_dir) if f.endswith('.png')])

for img_name in test_images:
    img_path = os.path.join(aliased_dir, img_name)
    aliased_img = Image.open(img_path).convert('L')

    aliased_tensor = transform(aliased_img).unsqueeze(0).to(device) # transform to PyTorch tensor

    with torch.no_grad():
        generated_tensor = generator(aliased_tensor).squeeze(0).cpu()
    print(f'Tensor shape: {generated_tensor.shape}')
    print(f'Tensor range: [{generated_tensor.min()}, {generated_tensor.max()}]')

    # convert to PIL
    generated_img = transforms.ToPILImage()(generated_tensor)
    print(f'PIL mode: {generated_img.mode}')
    print(f'PIL size: {generated_img.size}')

    # check background pixel
    bg_pixel = np.array(generated_img)[0, 0]
    print(f'Background pixel: {bg_pixel}')

    # save generated images
    output_path = os.path.join(output_dir, f'generated_{img_name}')
    generated_img.save(output_path)

    print(f'Saved: {output_path}')

print('Inference complete!')
