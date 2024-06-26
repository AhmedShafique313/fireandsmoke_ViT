from model import load_model
from tuning import fine_tune_model
from transformers import ViTFeatureExtractor, ViTForImageClassification
from data_setup import train_dataset, test_dataset, valid_dataset
from train import train_model
from predict import predict

def main():
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("One of the datasets is empty. Please check the data paths and ensure the datasets are correctly populated.")
    
    train_model(train_dataset.image_paths, train_dataset.labels, valid_dataset.image_paths, valid_dataset.labels)

if __name__ == "__main__":
    main()