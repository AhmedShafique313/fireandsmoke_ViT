from model import load_model
from tuning import fine_tune_model
from transformers import ViTFeatureExtractor, ViTForImageClassification
from data_setup import train_dataset, test_dataset, valid_dataset
# from train import train_model
# from predict import predict

def main():
    # Train the model
    train_model(train_dataset, valid_dataset)

    # Predict on a new image
    image_path = 'path/to/test/image.jpg'
    prediction = predict(image_path)
    print(f"Predicted class index: {prediction}")

if __name__ == "__main__":
    main()
