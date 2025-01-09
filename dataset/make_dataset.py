import os
import shutil
import random
import cv2
import numpy as np
import uuid

def copy_images(src_dir, dest_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Loop through all files and directories in the source directory
    for subdir_name in os.listdir(src_dir):
        subdir_path = os.path.join(src_dir, subdir_name)

        # Check if the subdirectory is actually a directory
        if os.path.isdir(subdir_path):
            # Look for image files in the subdirectory
            for filename in os.listdir(subdir_path):
                # Check if the file is an image file
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    src_file_path = os.path.join(subdir_path, filename)


                    # Generate a unique identifier
                    unique_id = uuid.uuid4().hex[:6]

                    # Construct the destination filename with the unique identifier
                    dest_filename = f"{subdir_name}_{os.path.splitext(filename)[0]}{os.path.splitext(filename)[1]}"
                    dest_file_path = os.path.join(dest_dir, dest_filename)

                    # Copy the image file to the destination directory
                    shutil.copy(src_file_path, dest_file_path)
                    print(f"Copied: {src_file_path} to {dest_file_path}")


def copy_random_images(source_folder, target_folder, percentage=10):
    """
    Selects a random percentage of images from a source folder and copies them to a target folder.

    Parameters:
    - source_folder: str, the path to the folder containing the original images.
    - target_folder: str, the path to the folder where selected images will be copied.
    - percentage: int, the percentage of images to select randomly. Default is 10.
    """
    # Clear existing files in the target folder
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)

    # Get a list of files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Calculate the number of files to select
    num_files_to_select = max(1, len(files) * percentage // 100)

    # Randomly select the files
    selected_files = random.sample(files, num_files_to_select)

    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Copy the selected files to the target folder
    for file in selected_files:
        source_path = os.path.join(source_folder, file)
        target_path = os.path.join(target_folder, file)
        shutil.copy2(source_path, target_path)  # Use copy2 to preserve metadata
    print(f"Copied {len(selected_files)} files to {target_folder}")

    # for file in selected_files:
    #     source_path = os.path.join(source_folder, file)
    #     os.remove(source_path)
    # print(f"Deleted {len(selected_files)} files from {source_folder}")

def rotate_and_save_images(source_folder, destination_folder, angle=90):
    # Iterate through all files in the source folder
    os.makedirs(destination_folder, exist_ok=True)
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Load an image
            image = cv2.imread(os.path.join(source_folder, filename))

            # Get image dimensions
            height, width = image.shape[:2]

            # Calculate the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

            # Apply the rotation transformation
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

            # Save the images
            # cv2.imwrite(os.path.join(destination_folder, 'original_' + filename), image)
            cv2.imwrite(os.path.join(destination_folder,filename), rotated_image)


def apply_color_augmentation_and_save_folder(input_folder, output_folder, brightness_factor=0.5,
                                             contrast_factor=1.5, saturation_factor=1.5, hue_factor=20):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Load an image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Apply color space augmentation
            augmented_image = image.astype(float)

            # Adjust brightness
            augmented_image[:, :, :] *= brightness_factor

            # Adjust contrast
            augmented_image[:, :, 0] = np.clip(augmented_image[:, :, 0] * contrast_factor, 0, 255)
            augmented_image[:, :, 1] = np.clip(augmented_image[:, :, 1] * contrast_factor, 0, 255)
            augmented_image[:, :, 2] = np.clip(augmented_image[:, :, 2] * contrast_factor, 0, 255)

            # Adjust saturation
            augmented_image[:, :, 0] = np.clip(augmented_image[:, :, 0] * saturation_factor, 0, 255)
            augmented_image[:, :, 1] = np.clip(augmented_image[:, :, 1] * saturation_factor, 0, 255)
            augmented_image[:, :, 2] = np.clip(augmented_image[:, :, 2] * saturation_factor, 0, 255)

            # Adjust hue
            augmented_image[:, :, 0] = (augmented_image[:, :, 0] + hue_factor) % 256

            # Convert back to uint8
            augmented_image = np.uint8(augmented_image)

            # Save the augmented image
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_folder, name + ext)
            cv2.imwrite(output_path, augmented_image)


def add_gaussian_noise_and_save_images(input_directory, destination_directory, mean=0, std=25):
    """
    Add Gaussian noise to images in the input directory and save them to the destination directory.

    Parameters:
        input_directory (str): Path to the directory containing input images.
        destination_directory (str): Path to the directory where noisy images will be saved.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    """
    # Create destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)

    # Get list of image files in the input directory
    image_files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

    # Process each image
    for filename in image_files:
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Load image
            image_path = os.path.join(input_directory, filename)
            image = cv2.imread(image_path)

            # Add Gaussian noise
            noise = np.random.normal(mean, std, image.shape)
            noisy_image = image + noise
            noisy_image = np.clip(noisy_image, 0, 255)
            noisy_image = noisy_image.astype(np.uint8)

            # Save noisy image
            name, ext = os.path.splitext(filename)
            destination_path = os.path.join(destination_directory, name + ext)
            cv2.imwrite(destination_path, noisy_image)


def split_train_test(split_ratio, base_dir, train_dir, test_dir):
    """
    Split the subdirectories containing images into train and test sets and save them in specified directories.

    Args:
        split_ratio (float): Ratio of train set size to total data size.
        base_dir (str): Path to the directory containing subdirectories with images.
        train_dir (str): Path to the directory where train set will be saved.
        test_dir (str): Path to the directory where test set will be saved.
    """
    # Create directories for train and test sets
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate over subdirectories
    for class_folder in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_folder)
        if os.path.isdir(class_path):
            # Create corresponding class directories in train and test sets
            train_class_dir = os.path.join(train_dir, class_folder)
            test_class_dir = os.path.join(test_dir, class_folder)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # Get list of image files in the subdirectory
            files = [f for f in os.listdir(class_path) if
                     f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
            # Shuffle the files to ensure randomness
            random.shuffle(files)

            # Split the files into train and test sets based on the ratio
            num_train = int(len(files) * split_ratio)
            train_files = files[:num_train]
            test_files = files[num_train:]

            # Copy files to train and test directories
            for file_name in train_files:
                src = os.path.join(class_path, file_name)
                dst = os.path.join(train_class_dir, file_name)
                shutil.copyfile(src, dst)

            for file_name in test_files:
                src = os.path.join(class_path, file_name)
                dst = os.path.join(test_class_dir, file_name)
                shutil.copyfile(src, dst)


# Specify the source directory containing the images
source_directory = "/home/fahadk/Project/FCC/dataset/food-101/normal/anomaly"
# source_directory = "/home/fahad/Project/CIPER/dataset/image/fruitsdataset/test"

# Specify the destination directory where images will be copied
destination_directory = "/home/fahadk/Project/FCC/dataset/food-101/normal/temp"
# Specify the destination directory where anamolous images will be saved
anomolous_directory = "/home/fahadk/Project/FCC/dataset/food-101/anomaly/"

# copy_images(source_directory,destination_directory)
# copy_random_images(source_directory, destination_directory)
# Call the function to copy images from source directory to destination directory
# rotate_and_save_images(destination_directory, anomolous_directory)
# apply_color_augmentation_and_save_folder(destination_directory, anomolous_directory)
# add_gaussian_noise_and_save_images(destination_directory, anomolous_directory)
# Call the function to split the images into test and train
# split_train_test(0.7,base_dir=anomolous_directory,train_dir=anomolous_directory+'train',test_dir=anomolous_directory+'test')




