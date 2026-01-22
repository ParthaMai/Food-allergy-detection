import pandas as pd

# Load Indian food ingredients dataset
df = pd.read_csv("indian_food_ingredients.csv")

# Retrieve ingredients for a given food name
def get_ingredients(food_name):
    result = df[df["Food_Name"].str.lower() == food_name.lower()]
    if not result.empty:
        return result.iloc[0]["Ingredients"]
    else:
        return None

