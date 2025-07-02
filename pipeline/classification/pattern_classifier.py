import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from PIL import Image

# --- Configuration ---
# IMPORTANT: Change this path to your dataset directory.
# The directory should contain subdirectories for each class, e.g., /data/cats_and_dogs/cat, /data/cats_and_dogs/dog
DATA_PATH = '../../data/classification'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- Feature Extraction Function ---

def extract_features(data_loader, model):
    """
    Extracts deep features from a dataset using a pre-trained model.
    """
    features_list = []
    labels_list = []

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations to speed up the process and save memory
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(DEVICE)
            # Get the feature vectors from the model
            outputs = model(inputs)

            # Move data to CPU and convert to NumPy arrays
            features_list.extend(outputs.cpu().numpy())
            labels_list.extend(targets.cpu().numpy())

    return np.array(features_list), np.array(labels_list)


# --- Prediction Function ---

def predict_image_probabilities(image_path, feature_extractor, classifier, class_names):
    """
    Takes an image path, processes it, and returns a vector of class probabilities.
    """
    # Define a specific transformation pipeline for inference (no random augmentation)
    inference_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None

    # Apply transformations and add a batch dimension
    image_tensor = inference_transforms(image).unsqueeze(0).to(DEVICE)

    # 1. Extract features using the feature extractor (e.g., ResNet)
    feature_extractor.eval()
    with torch.no_grad():
        features = feature_extractor(image_tensor).cpu().numpy()

    # 2. Get probability vector from the classifier (e.g., Logistic Regression)
    probabilities = classifier.predict_proba(features)[0]

    # Map probabilities to class names
    probability_vector = {class_names[i]: prob for i, prob in enumerate(probabilities)}
    return probability_vector


# --- Plotting Functions ---

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots a confusion matrix using seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def plot_decision_boundary(X_test_features, y_test, class_names):
    """
    Reduces feature dimensionality using PCA and plots decision boundaries
    with custom colors for specific classes.
    Note: This trains a new classifier on 2D data for visualization purposes only.
    """
    print("\nGenerating decision boundary plot with custom colors...")
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    X_test_2d = pca.fit_transform(X_test_features)

    # Train a simple classifier on the 2D data FOR VISUALIZATION ONLY
    classifier_2d = LogisticRegression(solver='liblinear')
    classifier_2d.fit(X_test_2d, y_test)

    # --- Custom Color Mapping ---
    # Define your desired colors for specific classes
    color_map = {
        'deserts': 'yellow',
        'forests': 'green',
        'patterns': 'red'
    }
    # Create a list of colors in the same order as class_names
    # Use grey as a default for any other classes
    custom_colors = [color_map.get(name, 'grey') for name in class_names]
    custom_cmap = ListedColormap(custom_colors)
    # --- End of Custom Color Mapping ---

    # Create a mesh grid
    x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
    y_min, y_max = X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict on the mesh grid
    Z = classifier_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundaries
    plt.figure(figsize=(12, 8))
    # Use the custom colormap for the background
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=custom_cmap)

    # Plot the test data points, also using the custom colormap
    scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test,
                          s=25, edgecolor='k', cmap=custom_cmap)

    plt.title('Decision Boundary Visualization (after PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Create a legend with correct colors
    # The legend handles mapping colors from the cmap automatically
    if len(class_names) <= 10:
        legend_elements = scatter.legend_elements()
        plt.legend(legend_elements[0], class_names, title="Classes")

    plt.show()

# --- Main Execution Logic ---

def main():
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"Error: Path '{DATA_PATH}' does not exist or is empty.")
        print("Please ensure the path is correct and contains class subdirectories with images.")
        return

    # --- Step 1: Data Augmentation and Preparation ---
    # Define transformations: resize, random augmentations, and normalization
    # Normalization values are standard for models pre-trained on ImageNet
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load data from folders
    image_dataset = datasets.ImageFolder(DATA_PATH, data_transforms)
    class_names = image_dataset.classes
    print(f"Found classes: {class_names}")
    print(f"Total number of images: {len(image_dataset)}")

    # Create a DataLoader to feed images in batches
    data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=32, shuffle=True)

    # --- Step 2: Load ResNet50 as a Feature Extractor ---
    # Load the pre-trained ResNet50 model
    feature_extractor_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Remove the final classification layer (fully connected)
    # This makes the model output a 2048-dimensional feature vector
    feature_extractor_model.fc = nn.Identity()
    feature_extractor_model = feature_extractor_model.to(DEVICE)

    # --- Step 3: Feature Extraction ---
    print("\nStarting feature extraction from images...")
    features, labels = extract_features(data_loader, feature_extractor_model)
    print(f"Extraction complete. Feature vector shape: {features.shape}")

    # --- Step 4: Train a Classifier ---
    # Split features into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    print("\nTraining the classifier (Logistic Regression)...")
    # Using a simple yet effective classifier
    # Increased max_iter to ensure convergence
    classifier = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
    classifier.fit(X_train, y_train)
    print("Training complete.")

    # --- Step 5: Evaluate the Model ---
    print("\n--- Evaluation on the Test Set ---")
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- Step 6: Generate Informational Plots ---
    # Plot 1: Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, class_names)

    # Plot 2: Decision Boundary (on 2D data)
    plot_decision_boundary(X_test, y_test, class_names)

    # --- Step 7: Predict on a new, single image ---
    print("\n--- Prediction on a Single Image ---")
    # !!! IMPORTANT: Replace this with the actual path to your image !!!
    test_image_path = "path/to/your/image.jpg"

    if os.path.exists(test_image_path):
        probabilities = predict_image_probabilities(test_image_path, feature_extractor_model, classifier, class_names)
        if probabilities:
            print(f"Predicted probabilities for '{test_image_path}':")
            for class_name, prob in probabilities.items():
                print(f"  - {class_name}: {prob:.4f}")
            # Find and print the class with the highest probability
            predicted_class = max(probabilities, key=probabilities.get)
            print(f"\n--> Most likely class: {predicted_class}")
    else:
        print(f"Skipping single image prediction because the file was not found at: '{test_image_path}'")
        print("Please update the 'test_image_path' variable with a valid path to run this step.")


if __name__ == '__main__':
    main()