import os
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

def augment_data(input_folder, output_folder, class_name, num_augmented_images=500, seed=42):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create an ImageDataGenerator with desired augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # List image files in the input folder
    image_files = [file for file in os.listdir(input_folder) if file.endswith(('.jpg', '.JPG', '.jpeg', '.png', '.gif'))]

    # Select random images for augmentation with a fixed seed for reproducibility
    np.random.seed(seed)
    selected_images = np.random.choice(image_files, num_augmented_images)
    print(len(image_files))
    print(len(selected_images))

    for filename in selected_images:
        input_path = os.path.join(input_folder, filename)
        output_base = f'{class_name}_aug_{os.path.splitext(filename)[0]}'

        # Open and convert the image to RGB mode
        with Image.open(input_path) as img:
            rgb_img = img.convert('RGB')

        # Convert the image to a numpy array
        img_array = np.array(rgb_img)

        # Reshape the image to (1, height, width, channels) as required by ImageDataGenerator
        img_array = img_array.reshape((1,) + img_array.shape)

        # Generate augmented images and save them
        i = 0
        for batch in datagen.flow(img_array, save_to_dir=output_folder, save_prefix=output_base, save_format='jpeg', seed=seed):
            i += 1
            
            if i >= num_augmented_images:
                break

if __name__ == "__main__":
    # Replace these paths with the actual paths to your original image folders
    original_yes_folder = 'normalized_dataset/train/yes'
    original_no_folder = 'normalized_dataset/train/no'

    # Replace these paths with the desired output paths for augmented images
    augmented_yes_folder = 'normalized_dataset/train_set/yes'
    augmented_no_folder = 'normalized_dataset/train_set/no'

    # You can adjust the number of augmented images per original image by changing num_augmented_images
    num_augmented_images = 6

    # Augment data for the 'Yes' class
    augment_data(original_yes_folder, augmented_yes_folder, 'yes', num_augmented_images)

    # Augment data for the 'No' class
    augment_data(original_no_folder, augmented_no_folder, 'no', num_augmented_images)
