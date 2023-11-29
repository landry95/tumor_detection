import os
import matplotlib.pyplot as plt


################ VISUALISE THE DATA BEFORE SPLITING ##########################
# Function to count files in a folder
def count(folder):
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

# Function to count files in yes and no folders
def count_category(yes_folder, no_folder):
    yes_count = count(yes_folder)
    no_count = count(no_folder)
    return yes_count, no_count

# Function to visualize the distribution of images in 'yes' and 'no' folders
def visualize_image_distribution(yes_folder, no_folder):
    # Get counts for each folder
    yes_count, no_count = count_category(yes_folder, no_folder)

    # Pie chart labels and counts
    labels = [f'Yes (Cancer): {yes_count}', f'No (No Cancer): {no_count}']
    sizes = [yes_count, no_count]
    colors = ['lightcoral', 'lightskyblue']

    # Create a pie chart
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Distribution of Images')
    plt.show()



################ VISUALISE THE DATA AFTER SPLITING ##########################
dataset_folder = 'dataset'

# Function to count files in a folder
def count_files(folder):
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

# Function to count files in yes and no folders within a set
def count_category_files(set_folder, category):
    category_folder = os.path.join(set_folder, category)
    return count_files(category_folder)

# Function to count files in train, validation, and test sets
def count_set_files(set_folder):
    yes_count = count_category_files(set_folder, 'yes')
    no_count = count_category_files(set_folder, 'no')
    return yes_count, no_count

# Function to visualize data
def visualize_data(dataset_folder=dataset_folder):
    # Paths to train, validation, and test sets
    train_folder = os.path.join(dataset_folder, 'train')
    val_folder = os.path.join(dataset_folder, 'validation')
    test_folder = os.path.join(dataset_folder, 'test')

    # Count images in each set
    train_counts = count_set_files(train_folder)
    val_counts = count_set_files(val_folder)
    test_counts = count_set_files(test_folder)

    # Bar chart labels and counts
    labels = ['Training Set', 'Validation Set', 'Test Set']
    yes_counts = [train_counts[0], val_counts[0], test_counts[0]]
    no_counts = [train_counts[1], val_counts[1], test_counts[1]]

    # Create a bar chart
    fig, ax = plt.subplots()
    width = 0.35
    bar1 = ax.bar(labels, yes_counts, width, label='Yes (Cancer)')
    bar2 = ax.bar(labels, no_counts, width, bottom=yes_counts, label='No (No Cancer)')

    # Annotate each bar with the counts
    for bar, count in zip(bar1 + bar2, yes_counts + no_counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Set labels and title
    ax.set_ylabel('Number of Images')
    ax.set_title('Distribution of Images in Train, Validation, and Test Sets')
    ax.legend()

    # Show the plot
    plt.show()
