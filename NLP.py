import pytesseract
from PIL import Image
from transformers import BertTokenizer, BertModel
import torch
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

model_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

def extract_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text.lower()

ALLERGENS = {
    "nuts": ["peanut", "almond", "cashew", "walnut", "brazil nut"],
    "dairy": ["milk", "whey", "butter", "lactose"],
    "gluten": ["wheat", "barley", "rye", "oats"],
    "soy": ["soy", "soybean", "lecithin"],
    "eggs": ["egg", "albumin"],
    "seafood": ["fish", "shrimp", "crab"],
    "preservatives": ["msg", "sodium benzoate"],
    "others": []
}

def analyze_allergens(text):
    detected = {}

    for category, keywords in ALLERGENS.items():
        found = []
        for word in keywords:
            if re.search(rf"\b{word}\b", text):
                found.append(word)
        detected[category] = found

    return detected


def filter_detected_allergens(allergens_dict):
    return {k: v for k, v in allergens_dict.items() if v}


if __name__ == "__main__":

    image_path = "dairymilk.jpg"


    text = extract_text(image_path)
    # print(text)

    found_allergens = analyze_allergens(text)

    filtered_allergens = filter_detected_allergens(found_allergens)
    print("\nDetected allergens in the label:")
    print(filtered_allergens)

