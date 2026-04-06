import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import pandas as pd

# useful functions

# normalised mean square error
def nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

# apply least squares brightness scaling
def brightness_correct(gen, gt):
    a = np.sum(gt * gen) / np.sum(gen ** 2)
    corrected = np.clip(a * gen, 0, 1)
    return corrected, a


# path setup

# generated images - change before running
generated_dir = 'generated_outputs/version-9'

# input aliased images - change before running
input_dir = 'datasets/OCMR/AL/R=8/test'

# ground truth images
gt_dir = 'datasets/OCMR/GT/test'

# sort generated images
gen_prefix = 'generated_'
generated_filenames = sorted([
    f for f in os.listdir(generated_dir)
    if f.endswith('.png') and f.startswith(gen_prefix)
])

results = []

# main eval loop
for gen_filename in generated_filenames:

    base_filename = gen_filename[len(gen_prefix):]
    gt_filename = base_filename.replace('_AL', '_GT')

    gen_path = os.path.join(generated_dir, gen_filename)
    input_path = os.path.join(input_dir, base_filename)
    gt_path = os.path.join(gt_dir, gt_filename)

    try:
        gen_img = Image.open(gen_path).convert('L')
        input_img = Image.open(input_path).convert('L')
        gt_img = Image.open(gt_path).convert('L')
    except FileNotFoundError:
        print(f'Missing image: {base_filename}')
        continue

    # convert to NumPy arrays and normalise
    gen_np = np.array(gen_img, dtype=np.float32) / 255.0
    input_np = np.array(input_img, dtype=np.float32) / 255.0
    gt_np = np.array(gt_img, dtype=np.float32) / 255.0

    # raw metric results
    mse_raw = mean_squared_error(gt_np, gen_np)
    nmse_raw = nmse(gt_np, gen_np)
    psnr_raw = peak_signal_noise_ratio(gt_np, gen_np, data_range=1.0)
    ssim_raw = structural_similarity(gt_np, gen_np, data_range=1.0)

    # call brightness correction function
    gen_corrected, scale_factor = brightness_correct(gen_np, gt_np)

    # brightness corrected metric results
    mse_corr = mean_squared_error(gt_np, gen_corrected)
    nmse_corr = nmse(gt_np, gen_corrected)
    psnr_corr = peak_signal_noise_ratio(gt_np, gen_corrected, data_range=1.0)
    ssim_corr = structural_similarity(gt_np, gen_corrected, data_range=1.0)

    # store results
    results.append({
        'Filename': base_filename,
        'ScaleFactor': scale_factor,
        'MSE_Raw': mse_raw,
        'NMSE_Raw': nmse_raw,
        'PSNR_Raw': psnr_raw,
        'SSIM_Raw': ssim_raw,
        'MSE_Corrected': mse_corr,
        'NMSE_Corrected': nmse_corr,
        'PSNR_Corrected': psnr_corr,
        'SSIM_Corrected': ssim_corr
    })

    # display metric results for each image in the test set
    plt.figure(figsize=(14, 5))
    plt.suptitle(
        f'{base_filename}\n'
        f'Raw → MSE:{mse_raw:.5f} | NMSE:{nmse_raw:.5f} | PSNR:{psnr_raw:.2f} dB | SSIM:{ssim_raw:.4f}\n'
        f'Corrected → MSE:{mse_corr:.5f} | NMSE:{nmse_corr:.5f} | PSNR:{psnr_corr:.2f} dB | SSIM:{ssim_corr:.4f} | Scale:{scale_factor:.3f}',
        fontsize=11
    )

    plt.subplot(1, 4, 1)
    plt.imshow(input_np, cmap='gray')
    plt.title('Input (Aliased)')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(gen_np, cmap='gray')
    plt.title('Generated (Raw)')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(gen_corrected, cmap='gray')
    plt.title('Generated (Corrected)')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(gt_np, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    plt.show()

# saved results of every test image in a csv file
if results:
    df = pd.DataFrame(results)
    csv_path = 'cgan_metrics_with_brightness_correction.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n Metrics saved to {csv_path}')
    print(df.describe().T)
else:
    print('No valid image pairs found.')
