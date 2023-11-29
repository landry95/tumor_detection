import os
import random
from shutil import copyfile

def split_dataset(original_dataset_path, new_dataset_path, validation_percentage=20, test_percentage=10):
    # Create the train, validation, and test directories
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(new_dataset_path, split), exist_ok=True)

    # Loop through each class in the original dataset
    for class_name in os.listdir(original_dataset_path):
        class_path = os.path.join(original_dataset_path, class_name)
        files = os.listdir(class_path)

        # Calculate the number of validation and test samples based on the percentages
        num_validation = int(len(files) * (validation_percentage / 100.0))
        num_test = int(len(files) * (test_percentage / 100.0))

        # Randomly shuffle the files
        random.shuffle(files)

        # Split the files into training, validation, and test sets
        train_files, validation_files, test_files = files[num_validation + num_test:], files[:num_validation], files[num_validation:num_validation + num_test]

        # Loop through the splits and copy files
        for split, split_files in zip(['train', 'validation', 'test'], [train_files, validation_files, test_files]):
            split_path = os.path.join(new_dataset_path, split, class_name)
            os.makedirs(split_path, exist_ok=True)
            
            # Copy files to the new structure
            for file in split_files:
                copyfile(os.path.join(class_path, file), os.path.join(split_path, file))

    print("Dataset split into train, validation, and test sets.")


original_dataset_path = 'data'
new_dataset_path = 'dataset'

# Call the function to split the dataset
split_dataset(original_dataset_path, new_dataset_path)
