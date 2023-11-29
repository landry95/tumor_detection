from PIL import Image
import os

def resize_images(input_folder, output_folder, target_size=(224, 224)):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List subdirectories ('yes' and 'no') in the input folder
    subdirectories = [subdir for subdir in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, subdir))]

    for subdir in subdirectories:
        input_subdir = os.path.join(input_folder, subdir)
        output_subdir = os.path.join(output_folder, subdir)

        # Create output subdirectory if it doesn't exist
        os.makedirs(output_subdir, exist_ok=True)

        # Iterate over image files in the input subdirectory
        for filename in os.listdir(input_subdir):
            input_path = os.path.join(input_subdir, filename)
            output_path = os.path.join(output_subdir, filename)

            # Open and resize the image
            with Image.open(input_path) as img:
                # Convert the image to RGB mode
                rgb_img = img.convert('RGB')
                resized_img = rgb_img.resize(target_size, Image.ANTIALIAS)

            # Save the resized image to the output subdirectory
            resized_img.save(output_path)

if __name__ == "__main__":
    # Replace these paths with the actual paths to your train, validation, and test folders
    train_folder = 'dataset/train'
    val_folder = 'dataset/validation'
    test_folder = 'dataset/test'

    # Replace these paths with the desired output paths for resized images
    resized_train_folder = 'resized_dataset/train'
    resized_val_folder = 'resized_dataset/validation'
    resized_test_folder = 'resized_dataset/test'

    # You can run this script for each set (train, validation, test) separately
    resize_images(train_folder, resized_train_folder)
    resize_images(val_folder, resized_val_folder)
    resize_images(test_folder, resized_test_folder)

