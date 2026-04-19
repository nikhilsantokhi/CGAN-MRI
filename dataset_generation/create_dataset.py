import read_ocmr as read
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from ismrmrdtools import transform
from scipy.ndimage import zoom

def process_mri_data(file_path):
    # Step 1: Load data
    kData, param = read.read_ocmr(file_path)
    print('Dimension of kData: ', kData.shape)

    #dim_kData = kData.shape

    # Step 2: Average the k-space data accumulations
    kData_tmp = np.mean(kData, axis = 8)

    # Step 3: Apply IFFT to transform k-space into the spatial domain
    im_coil = transform.transform_kspace_to_image(kData_tmp, [0, 1]) # IFFT (2D image)

    # Step 4: Resize MR image tensor for real and imaginary parts separately
    # [kx, ky, kz, coil, frame, set, slice, rep ,avg]

    # squeeze so it isn't needed in both generate image functions
    im_coil = np.squeeze(im_coil[:,:,:,:,:,:,:,:])
    coil = im_coil.shape[2]
    frame = im_coil.shape[3]

    # new_size = (128, 128, coil, frame)
    new_size = (256, 256, coil, frame)

    resized_image_tensor = complex_mri_resize(im_coil, new_size)

    # Step 5: Generate aliased MR images
    aliased_images = generate_aliased_images(resized_image_tensor)

    # Step 6: Generate fully sampled ground-truth MR images
    fully_sampled_images = generate_fully_sampled_images(resized_image_tensor)

    return aliased_images, fully_sampled_images

def complex_mri_resize(image_tensor, new_size):
        scale_factors = np.array(new_size) / np.array(image_tensor.shape)
        real_resized = zoom(np.real(image_tensor), scale_factors, order=3)
        imag_resized = zoom(np.imag(image_tensor), scale_factors, order=3)

        return real_resized + 1j * imag_resized

first_display_done = False

def generate_aliased_images(resized_image_tensor, centre_fraction=0.08):
    global first_display_done

    # Step 5.1: Transform resized image tensor back to the k-space
    resized_kspace_tensor = transform.transform_image_to_kspace(resized_image_tensor, [0, 1])

    height, width, num_coils, frames = resized_kspace_tensor.shape

    # Step 5.2: Generate Cartesian binary sampling mask (R=2, change for other accelerations)
    binary_mask = np.zeros_like(resized_kspace_tensor)

    # Preserve centre fraction
    centre_size = int(height * centre_fraction)
    centre_start = (height - centre_size) // 2
    centre_end = centre_start + centre_size

    # Undersample rows
    binary_mask[::8, :, :] = 1
    binary_mask[centre_start:centre_end, :, :] = 1

    # Undersample columns
    # binary_mask[:, ::2, :] = 1
    # binary_mask[:, centre_start:centre_end, :] = 1

    # Step 5.3: Undersample the resized_kspace using the binary mask
    US_kspace = resized_kspace_tensor * binary_mask

    # Step 5.4: Apply IFFT to get the under-sampled MR images
    US_MR_image_tensor = transform.transform_kspace_to_image(US_kspace, [0, 1])

    # Step 5.5: Merge the different channels via the SoS procedure
    im_sos_full = np.sqrt(np.sum(np.abs(US_MR_image_tensor) ** 2, 2))

    aliased_images = im_sos_full

    # Display k-space, mask and masked k-space of the first file once
    if not first_display_done:
        mid_coil = num_coils // 2

        kspace_mag = np.log1p(np.abs(resized_kspace_tensor[:, :, mid_coil]))
        mask_vis = np.abs(binary_mask[:, :, mid_coil])
        masked_kspace_mag = np.log1p(np.abs(US_kspace[:, :, mid_coil]))

        plt.figure(figsize= (12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(kspace_mag[:, :, 0], cmap='gray')
        plt.title('Original K-space')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask_vis[:, :, 0], cmap='gray')
        plt.title('Binary Mask (with centre)')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(masked_kspace_mag[:, :, 0], cmap='gray')
        plt.title('Masked K-space')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        first_display_done = True

    return aliased_images

def generate_fully_sampled_images(resized_image_tensor):
    # Step 6.1: Merge the different channels
    im_sos_full = np.sqrt(np.sum(np.abs(resized_image_tensor) ** 2, 2))

    fully_sampled_images = im_sos_full

    return fully_sampled_images

def save_images(image_array, base_name, prefix, save_dir="C:\Python\Python Projects\OCMR\Python\Acceleration Datasets\R=8"):
        # Create output directory if not exists
        output_dir = os.path.join(save_dir, prefix)
        os.makedirs(output_dir, exist_ok=True)

        # Save each image looping through the slices
        for i in range(image_array.shape[2]):
            # Extract the 2D image for the current slice
            img = image_array[:, :, i]

            # Normalise the array
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img_normalised = img.astype(np.uint8)

            # Save image as a PNG to disk
            filename = f"{base_name}_{prefix}_{i}.png"
            file_path = os.path.join(output_dir, filename)
            cv2.imwrite(file_path, np.transpose(img_normalised))

        print(f"Saved {int(image_array.shape[2])} {prefix} images to: {output_dir}")