import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img = cv2.resize(img, (128, 128))  # Resize for consistency
        images.append(img)
        labels.append(label)
    return images, labels

# Load both categories
first_print_images, first_print_labels = load_images("/content/dataset/First Print", 0)  # Label 0 for original
second_print_images, second_print_labels = load_images("/content/dataset/Second Print", 1)  # Label 1 for counterfeit

# Combine data
X = np.array(first_print_images + second_print_images)
y = np.array(first_print_labels + second_print_labels)

# Normalize pixel values to [0,1]
X = X / 255.0

import random

# Select random samples from both classes
first_print_samples = random.sample(first_print_images, 5)
second_print_samples = random.sample(second_print_images, 5)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("First Prints vs. Second Prints")

for i in range(5):
    axes[0, i].imshow(first_print_samples[i], cmap='gray')
    axes[0, i].set_title("First Print")
    axes[0, i].axis("off")

    axes[1, i].imshow(second_print_samples[i], cmap='gray')
    axes[1, i].set_title("Second Print")
    axes[1, i].axis("off")

plt.show()


def plot_histogram(image, title):
    plt.hist(image.ravel(), bins=256, range=[0, 256], alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

# Select one random image from each class
random_first_print = random.choice(first_print_images)
random_second_print = random.choice(second_print_images)

plot_histogram(random_first_print, "Pixel Intensity Distribution - First Print")
plot_histogram(random_second_print, "Pixel Intensity Distribution - Second Print")

def detect_edges(image):
    edges = cv2.Canny(image, 50, 150)  # Apply Canny edge detection
    return edges

first_edges = detect_edges(random_first_print)
second_edges = detect_edges(random_second_print)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(first_edges, cmap='gray')
ax[0].set_title("Edges - First Print")
ax[1].imshow(second_edges, cmap='gray')
ax[1].set_title("Edges - Second Print")
plt.show()

from skimage.filters import sobel

# Define dataset paths
first_prints_path = "/content/dataset/First Print"
second_prints_path = "/content/dataset/Second Print"

# Load images
first_prints = [cv2.imread(os.path.join(first_prints_path, f), cv2.IMREAD_GRAYSCALE)
                for f in os.listdir(first_prints_path)[:5]]
second_prints = [cv2.imread(os.path.join(second_prints_path, f), cv2.IMREAD_GRAYSCALE)
                 for f in os.listdir(second_prints_path)[:5]]

# Function to calculate image sharpness using Laplacian variance
def calculate_sharpness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Function to calculate edge definition using Sobel filter
def edge_sharpness(image):
    return cv2.Canny(image, 100, 200).mean()

# Function to calculate histogram variance (high variance means more noise)
def histogram_variance(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    return np.var(hist)

# Analyze first prints
first_sharpness = [calculate_sharpness(img) for img in first_prints]
first_edges = [edge_sharpness(img) for img in first_prints]
first_noise = [histogram_variance(img) for img in first_prints]

# Analyze second prints
second_sharpness = [calculate_sharpness(img) for img in second_prints]
second_edges = [edge_sharpness(img) for img in second_prints]
second_noise = [histogram_variance(img) for img in second_prints]

# Print statistics
print("🔹 **First Prints (Original)**")
print(f"  - Average Sharpness: {np.mean(first_sharpness):.2f}")
print(f"  - Average Edge Definition: {np.mean(first_edges):.2f}")
print(f"  - Average Noise Level: {np.mean(first_noise):.2f}\n")

print("🔹 **Second Prints (Counterfeit)**")
print(f"  - Average Sharpness: {np.mean(second_sharpness):.2f}")
print(f"  - Average Edge Definition: {np.mean(second_edges):.2f}")
print(f"  - Average Noise Level: {np.mean(second_noise):.2f}")

# Plot comparisons
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(["First Prints", "Second Prints"], [np.mean(first_sharpness), np.mean(second_sharpness)], color=['blue', 'red'])
axes[0].set_title("Image Sharpness")

axes[1].bar(["First Prints", "Second Prints"], [np.mean(first_edges), np.mean(second_edges)], color=['blue', 'red'])
axes[1].set_title("Edge Definition")

axes[2].bar(["First Prints", "Second Prints"], [np.mean(first_noise), np.mean(second_noise)], color=['blue', 'red'])
axes[2].set_title("Noise Level")

plt.show()
