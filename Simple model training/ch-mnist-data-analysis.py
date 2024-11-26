import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path
file_path = 'C:\\Users\\221AIG03\\OneDrive - 이화여자대학교\\Desktop\\VS codes\\ch-mnist\\hmnist_64_64_L.csv'
print("Loading dataset...")
data = pd.read_csv(file_path)

# Extract pixel data and labels from csv
pixel_columns = [col for col in data.columns if 'pixel' in col]
images = data[pixel_columns].values.reshape(-1, 64, 64)  # Reshape into 64x64 images
labels = data['label'].values

# Define categories (replace with appropriate labels)
categories = ["Tumor", "Stroma", "Complex", "Lympho", "Debris", "Mucosa", "Adipose", "Empty"]

# Visualize sample images (gray scale)
def plot_samples(images, labels, categories, samples_per_class=5):
    """Plot sample images for each category."""
    fig, axes = plt.subplots(len(categories), samples_per_class, figsize=(15, 10))
    for i, category in enumerate(categories):
        category_indices = np.where(labels == i)[0]
        for j in range(samples_per_class):
            if j < len(category_indices):
                ax = axes[i, j]
                ax.imshow(images[category_indices[j]], cmap='gray')
                ax.axis('off')
                if j == 0:
                    ax.set_title(category, fontsize=12)
    plt.tight_layout()
    plt.show()

print("Visualizing sample images...")
plot_samples(images, labels, categories)

# Function to plot class distribution
def plot_distribution(labels, categories):
    """Plot the class distribution."""
    sns.countplot(x=labels, palette='viridis')
    plt.xticks(range(len(categories)), categories, rotation=45)
    plt.title('Class Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.show()

print("Plotting class distribution...")
plot_distribution(labels, categories)

# Function to analyze pixel intensity distribution
def plot_pixel_distribution(images):
    """Plot the pixel intensity distribution."""
    flattened_pixels = images.flatten()
    plt.hist(flattened_pixels, bins=50, color='blue', alpha=0.7)
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

print("Plotting pixel intensity distribution...")
plot_pixel_distribution(images)
