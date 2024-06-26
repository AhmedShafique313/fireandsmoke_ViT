from PIL import Image
import torch
from transformers import ViTFeatureExtractor

# creating a custom class for the fire and smoke
class SmokeFireDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, feature_extractor):
        self.image_paths = image_paths
        self.labels = labels
        self.feature_extractor = feature_extractor

# creating class for returns number of samples

def __len__(self):
    return len(self.image_paths)

def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs['labels'] = torch.tensor(label)
        return inputs


def preprocess_image(image_path, feature_extractor):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs
