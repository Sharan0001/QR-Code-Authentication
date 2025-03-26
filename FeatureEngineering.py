import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import laplace
from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def load_images(folder, label):
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(label)
    return images, labels

# Load dataset
first_print_images, first_print_labels = load_images("/content/dataset/First Print", 0)
second_print_images, second_print_labels = load_images("/content/dataset/Second Print", 1)
X = np.array(first_print_images + second_print_images)
y = np.array(first_print_labels + second_print_labels)
X = X / 255.0  # Normalize pixel values

def extract_features(images):
    feature_vectors = []
    for img in images:
        img_uint8 = (img * 255).astype(np.uint8)

        # LBP features
        lbp = local_binary_pattern(img_uint8, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

        # GLCM features
        glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
        glcm_correlation = graycoprops(glcm, 'correlation')[0, 0]
        glcm_energy = graycoprops(glcm, 'energy')[0, 0]
        glcm_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        # Blurriness
        blurriness = cv2.Laplacian(img_uint8, cv2.CV_64F).var()

        # Edge density
        edges = cv2.Canny(img_uint8, 100, 200)
        edge_density = np.sum(edges) / (128 * 128)

        # Shannon Entropy
        entropy = shannon_entropy(img_uint8)

        # Fourier Variance
        f_transform = np.fft.fft2(img_uint8)
        f_transform_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_transform_shift)
        fourier_variance = np.var(magnitude_spectrum)

        features = np.hstack([lbp_hist, glcm_contrast, glcm_correlation, glcm_energy,
                              glcm_homogeneity, blurriness, edge_density, entropy, fourier_variance])
        feature_vectors.append(features)
    return np.array(feature_vectors)

# Extract features
X_features = extract_features(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y)

# Feature Importance Analysis using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
feature_ranks = np.argsort(importances)[::-1]
selected_features = feature_ranks[:10]

# Feature names
feature_names = ["LBP_" + str(i) for i in range(10)] + ["GLCM_Contrast", "GLCM_Correlation", "GLCM_Energy", "GLCM_Homogeneity", "Blurriness", "Edge_Density", "Entropy", "Fourier_Variance"]
selected_feature_names = [feature_names[i] for i in selected_features]

# Train SVM on selected features
svm_model = SVC(kernel='rbf', C=1000, gamma='scale')
svm_model.fit(X_train[:, selected_features], y_train)
y_pred = svm_model.predict(X_test[:, selected_features])

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize Feature Importance
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

print("Feature Ranking:\n", feature_importance_df)

plt.figure(figsize=(10, 5))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis")
plt.title("Feature Importance Ranking")
plt.show()
