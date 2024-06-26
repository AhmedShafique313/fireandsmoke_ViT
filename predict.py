from model import load_model
from data_preprocessing import preprocess_image

def predict(image_path):
    feature_extractor, model = load_model()
    inputs = preprocess_image(image_path, feature_extractor)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx
