from transformers import ViTFeatureExtractor, ViTForImageClassification

# load the pretrained ViT model
def load_model():
    feature_extraction = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    return feature_extraction, model
