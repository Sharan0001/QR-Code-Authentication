import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
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

def extract_features(images):
    feature_list = []
    for img in images:
        features = []

        # Local Binary Pattern (LBP)
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        features.extend(hist)

        # Laplacian Variance (Sharpness)
        laplacian_var = laplace(img).var()
        features.append(laplacian_var)

        # Shannon Entropy
        entropy = shannon_entropy(img)
        features.append(entropy)

        # GLCM Features
        glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        features.extend([contrast, correlation, energy, homogeneity])

        # Histogram of Oriented Gradients (HOG)
        hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        features.extend(hog_features[:20])  # Take first 20 values to reduce dimensionality

        feature_list.append(features)

    return np.array(feature_list)

# Load dataset
first_print_images, first_print_labels = load_images("/content/dataset/First Print", 0)
second_print_images, second_print_labels = load_images("/content/dataset/Second Print", 1)

X = np.array(first_print_images + second_print_images)
y = np.array(first_print_labels + second_print_labels)

# Extract features
X_features = extract_features(X)

# Feature Selection using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_features, y)
feature_importance = rf.feature_importances_
important_indices = np.argsort(feature_importance)[-10:]
X_selected = X_features[:, important_indices]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='rbf', C=1000, gamma='scale')
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly']
}

# Initialize SVM model
svm = SVC()

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and model
print("Best Parameters:", grid_search.best_params_)
best_svm = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Tuned SVM Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
