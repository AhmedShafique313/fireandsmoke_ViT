import os
from data_preprocessing import SmokeFireDataset
from model import load_model

# Define paths to your dataset folders
train_folder = r'C:\Users\Personal\Documents\projects\train'
test_folder = r'C:\Users\Personal\Documents\projects\test'
valid_folder = r'C:\Users\Personal\Documents\projects\valid'

# Helper function to get image paths and labels
def get_image_paths_and_labels(folder):
    image_paths = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for image_file in os.listdir(label_folder):
                if image_file.endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(label_folder, image_file))
                    labels.append(int(label))  # Assuming folder names are the class labels
    return image_paths, labels

# Get paths and labels for train, test, and validation sets
train_image_paths, train_labels = get_image_paths_and_labels(train_folder)
test_image_paths, test_labels = get_image_paths_and_labels(test_folder)
valid_image_paths, valid_labels = get_image_paths_and_labels(valid_folder)

# Load model and feature extractor
feature_extractor, model = load_model()

# Create datasets
train_dataset = SmokeFireDataset(train_image_paths, train_labels, feature_extractor)
test_dataset = SmokeFireDataset(test_image_paths, test_labels, feature_extractor)
valid_dataset = SmokeFireDataset(valid_image_paths, valid_labels, feature_extractor)
