import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pandas as pd

# brightness correct function
def brightness_correct(gen, gt):
    a = np.sum(gt * gen) / np.sum(gen ** 2)
    corrected = np.clip(a * gen, 0, 1)
    return corrected, a


# path setup

# generated images - change before running
generated_dir = 'generated_outputs/version-9'

# ground truth images
gt_dir = 'datasets/OCMR/GT/test'

# sort generated images
gen_prefix = 'generated_'
generated_filenames = sorted([
    f for f in os.listdir(generated_dir)
    if f.endswith('.png') and f.startswith(gen_prefix)
])


# store metric values
psnr_values, ssim_values, scale_factors, sample_indices, filenames = [], [], [], [], []

# main eval loop
for idx, gen_filename in enumerate(generated_filenames):
    base_filename = gen_filename[len(gen_prefix):]
    gt_filename = base_filename.replace('_AL', '_GT')

    gen_path = os.path.join(generated_dir, gen_filename)
    gt_path = os.path.join(gt_dir, gt_filename)

    try:
        gen_img = Image.open(gen_path).convert('L')
        gt_img = Image.open(gt_path).convert('L')
    except FileNotFoundError:
        print(f'Skipping {base_filename} (missing files)')
        continue

    # convert to NumPy arrays and normalize
    gen_np = np.array(gen_img, dtype=np.float32) / 255.0
    gt_np = np.array(gt_img, dtype=np.float32) / 255.0

    # apply brightness correction
    gen_corr, scale_factor = brightness_correct(gen_np, gt_np)

    # main evaluation metrics
    psnr_val = peak_signal_noise_ratio(gt_np, gen_corr, data_range=1.0)
    ssim_val = structural_similarity(gt_np, gen_corr, data_range=1.0)

    # store results
    filenames.append(base_filename)
    sample_indices.append(idx)
    scale_factors.append(scale_factor)
    psnr_values.append(psnr_val)
    ssim_values.append(ssim_val)

# convert to arrays
psnr_values = np.array(psnr_values)
ssim_values = np.array(ssim_values)
scale_factors = np.array(scale_factors)

# save results to csv file
df = pd.DataFrame({
    'Filename': filenames,
    'ScaleFactor': scale_factors,
    'PSNR': psnr_values,
    'SSIM': ssim_values
})
csv_path = 'psnr_ssim_brightness_corrected.csv'
df.to_csv(csv_path, index=False)
print(f'\nBrightness-corrected metrics saved to {csv_path}')
print(df.describe().T)

# plot two graphs showing PSNR and SSIM activity of all test images
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# PSNR plot
axes[0].plot(sample_indices, psnr_values, label='PSNR (dB)', color='tab:blue', marker='o', alpha=0.8)
axes[0].axhline(np.mean(psnr_values), color='tab:blue', linestyle='--', alpha=0.6,
                label=f'Mean PSNR: {np.mean(psnr_values):.2f} dB')
axes[0].set_ylabel('PSNR (dB)')
axes[0].set_title('PSNR Across Test Set (Brightness Corrected)')
axes[0].grid(True, linestyle='--', alpha=0.3)
axes[0].legend()

# SSIM plot
axes[1].plot(sample_indices, ssim_values, label='SSIM', color='tab:orange', marker='x', alpha=0.8)
axes[1].axhline(np.mean(ssim_values), color='tab:orange', linestyle='--', alpha=0.6,
                label=f'Mean SSIM: {np.mean(ssim_values):.4f}')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('SSIM')
axes[1].set_title('SSIM Across Test Set (Brightness Corrected)')
axes[1].grid(True, linestyle='--', alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()
