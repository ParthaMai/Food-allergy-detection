import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch




model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(
    model_name,
    use_fast=True
)
model.eval() 


df = pd.read_csv("indian_food_ingredients.csv")
FOOD_LABELS = df["Food_Name"].tolist()

def predict_food(image_path, labels):
    """
    Predict food name using CLIP (image-text similarity)
    """
    image = Image.open(image_path).convert("RGB")

    labels = [str(l) for l in labels]

    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Best match
    best_idx = probs.argmax().item()
    food_name = labels[best_idx]
    confidence = probs[0][best_idx].item()

    return food_name, confidence

