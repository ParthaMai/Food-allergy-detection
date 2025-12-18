from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import json

from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

@app.route('/test')
def test():
    return jsonify({"message": "API is running"})

@app.route("/")
def home():
    return "Food Allergy Detection API is running"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/Ingredients", methods=["POST"])
def Ingredients():
    try:
        if "image" not in request.files or request.files["image"].filename == "":
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)
        
        from Bert import extract_text
        text = extract_text(image_path)

        from Bert import normalize_text
        ingredients = normalize_text(text)

        # BERT
        from Bert import analyze_allergens
        allergens = analyze_allergens(ingredients)
        

        return jsonify({
            "ingredients": ingredients,
            "allergens": allergens
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/analyze", methods=["POST"])
def analyze():
    try:

        mode = request.form.get("mode", "food")
        exclude = request.form.get("exclude", "[]")
        exclude = json.loads(exclude)

        if "image" not in request.files or request.files["image"].filename == "":
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        # CLIP
        if mode == "food":
            from Clip import predict_food, FOOD_LABELS
            # Filter labels to exclude previous wrong predictions
            labels_to_search = [f for f in FOOD_LABELS if f not in exclude]
            food_name, confidence = predict_food(image_path, labels_to_search)
            if mode == "food":
                return jsonify({
                    "food": food_name,
                    "confidence": round(confidence, 3)
            })


        # Ingredients
        food_name = request.form.get("food_name")
        if not food_name:
            return jsonify({"error": "Food name missing"}), 400
        from ingredients import get_ingredients
        ingredients = get_ingredients(food_name)

        # BERT
        from Bert import analyze_allergens
        allergens = analyze_allergens(ingredients)
        

        return jsonify({
            "food": food_name,
            "ingredients": ingredients,
            "allergens": allergens
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
