from os import listdir
from create_dataset import process_mri_data, save_images

def main():
    files = listdir('D:/Datasets/OCMR_data/')
    no = 0
    print(len(files))
    for f in files:
        # Load the data, display size of kData and scan parameters
        file_path = 'D:/Datasets/OCMR_data/' + f
        # print(file_path)
        if f.startswith('fs'):
            print(f)
            no += 1
            print(no)
            aliased_images, fully_sampled_images = process_mri_data(file_path)
            new_file_name = f.replace('.h5', '')
            save_images(aliased_images, new_file_name, 'AL')
            save_images(fully_sampled_images, new_file_name, 'GT')


if __name__ == '__main__':
    main()