from model import load_model, fine_tune_model
# from tuning import fine_tune_model
from data_preprocessing import SmokeFireDataset


def train_model(train_image_paths, train_labels, val_image_paths, val_labels):
    feature_extractor, model = load_model()

    train_dataset = SmokeFireDataset(train_image_paths, train_labels, feature_extractor)
    val_dataset = SmokeFireDataset(val_image_paths, val_labels, feature_extractor)

    fine_tune_model(model, train_dataset, val_dataset)
