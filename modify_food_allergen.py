import pandas as pd

# Load CSV
df = pd.read_csv("food_data_alphabetically_sorted.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Clean columns
df["Food_Name"] = df["Food_Name"].fillna("").str.lower()
df["Ingredients"] = df["Ingredients"].fillna("").str.lower()


# Allergens to detect
allergens = [
    "peanut", "almond", "pistachio", "cashew",
    "milk", "butter", "cheese", "paneer", "ghee",
    "egg",
    "fish", "prawn",
    "moong dal", "chana dal",
    "tomato", "banana",
    "capsicum", "mushroom", "bitter gourd",
    "mustard",
    "chocolate",
    "chicken", "mutton"
]

# Create 0/1 allergen columns
for allergen in allergens:
    df[allergen] = (
        df["Ingredients"]
        .str.contains(allergen, case=False, na=False)
        .astype(int)
    )

result = df[["Food_Name", "Ingredients"] + allergens]

result.to_csv(
    "modified_final_food_allergen.csv",
    index=False
)

print(result.head())