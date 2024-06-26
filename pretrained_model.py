# pip install git+https://github.com/huggingface/transformers.git

# after that import required libraries and install them all 
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

image_path = 'test_image.jpg'
image1 = Image.open(image_path)
image1.show()

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = feature_extractor(images=image1, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])