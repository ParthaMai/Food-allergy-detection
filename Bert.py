import pytesseract
from PIL import Image
from transformers import BertTokenizer, BertModel
import torch
import re


import csv
from collections import defaultdict

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()   # inference mode

# Extract text from an image using OCR
def extract_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return normalize_text(text)


# Normalize text by lowercasing and removing special characters
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ALLERGENS = {
#     "nuts": ["peanut", "almond", "almonds", "cashew", "walnut", "brazil nut", "pistachio", "raisin"],
#     "dairy": ["milk", "whey", "butter", "lactose", "milk solids", "cheese", "ice cream", "milk based beverage", "paneer", "curd", "ghee"],
#     "gluten": ["wheat", "barley", "rye", "oats", "bread", "bread loaf", "roti", "pasta", "pizza", "maida", "attā"],
#     "eggs": ["hen egg", "duck egg", "quail egg", "goose egg", "pigeon egg", "albumin", "whole egg boiled"],
#     "seafood": ["fish", "prawn", "crab", "lobster", "squid", "mussels"],
#     "pulses": ["chickpeas", "moong dal", "chana dal", "kidney beans"],
#     "fruits": ["avocado", "blackberry", "blueberry", "dates", "pineapple", "strawberry", "tomato", "banana", "mango"],
#     "vegetables": ["capsicum", "eggplant", "spinach", "mushroom", "okra", "bitter gourd"],
#     "spices": ["mustard", "turmeric", "cumin", "coriander", "chili"],
#     "sweets": ["chocolate", "gulab jamun", "jalebi", "barfi", "ladoo"],
#     "meat": ["cooked meat", "raw meat", "chicken", "mutton", "goat"],
#     "beverages": ["alcohol", "alcohol glass", "non milk based beverage", "tea", "coffee"],
#     "preservatives": ["msg", "sodium benzoate"],
#     "seeds": ["sesame", "poppy seeds"]
# }

# Load allergen keywords from CSV file
def load_allergens(csv_path="allergens.csv"):
    allergens = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            allergens[row["category"]].append(row["keyword"])

    return dict(allergens)

ALLERGENS = load_allergens()




def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1)

# Analyze text to detect allergens using Exact keyword and semantic matching
def analyze_allergens(text, similarity_threshold=0.60):
    if text is None: 
        text = ""
    detected = {}

    text_embedding = get_embedding(str(text)) 
    for category, keywords in ALLERGENS.items():
        found = []

        for keyword in keywords:
            # Exact keyword match
            if re.search(rf"\b{re.escape(keyword)}\b", text):
                found.append(keyword)
                continue

            # BERT semantic validation
            keyword_embedding = get_embedding(str(keyword))
            similarity = torch.cosine_similarity(
                text_embedding, keyword_embedding
            ).item()

            if similarity >= similarity_threshold:
                found.append(keyword)

        if found:
            detected[category] = sorted(set(found))

    return detected
