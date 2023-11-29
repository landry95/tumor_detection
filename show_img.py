import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def visualize_img(dataset_folder='dataset', sample_size=20):
    # Paths to train, validation, and test sets
    train_folder = os.path.join(dataset_folder, 'train')

    # Function to get a list of sample images from a category
    def get_sample_images(category_folder, sample_size):
        # List all files in the category folder and take a sample
        images = os.listdir(category_folder)[:sample_size]
        return [os.path.join(category_folder, img) for img in images]

    # Get sample images from 'yes' and 'no' categories in the training set
    yes_samples = get_sample_images(os.path.join(train_folder, 'yes'), sample_size)
    no_samples = get_sample_images(os.path.join(train_folder, 'no'), sample_size)

    # Display sample images in a grid
    fig, axes = plt.subplots(2, sample_size, figsize=(15, 5))
    fig.suptitle('Sample Images from Training Set', fontsize=16)

    # Display 'Yes' sample images
    for i, img_path in enumerate(yes_samples):
        img = mpimg.imread(img_path)
        axes[0, i].imshow(img, cmap='gray')  # Explicitly set the colormap to 'gray'
        axes[0, i].axis('off')
        axes[0, i].set_title('Yes')

    # Display 'No' sample images
    for i, img_path in enumerate(no_samples):
        img = mpimg.imread(img_path)
        axes[1, i].imshow(img, cmap='gray')  # Explicitly set the colormap to 'gray'
        axes[1, i].axis('off')
        axes[1, i].set_title('No')

    # Show the plot
    plt.show()

