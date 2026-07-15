import pandas as pd

df = pd.read_csv("modified_final_food_allergen.csv")

print("Butter count:", df["butter"].sum())
print("Egg count:", df["egg"].sum())
print("Milk count:", df["milk"].sum())
print("Chocolate count:", df["chocolate"].sum())