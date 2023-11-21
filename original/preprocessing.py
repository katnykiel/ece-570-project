# Import libraries
import json
import pandas as pd
from pymatgen.core import Structure
from crystal_embed import visualize_structure
from PIL import Image
import numpy as np

# Read the JSON file into a dictionary
with open("structures.json", "r") as f:
    data = json.load(f)


# Convert this dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Print the total sum of the is_stable column
print(df["stables"].sum())

# Convert the structure column into a pymatgen Structure object
df["structure"] = df["structures"].apply(lambda x: Structure.from_dict(x))

# Save the first n images to file as a single nxn image
images = []
n = 8
for i in range(n**2 + 2):
    if i == 5 or i == 7:
        continue
    data = visualize_structure(df["structure"][i])
    image = Image.fromarray(data[0].astype(np.uint8))
    images.append(image)

# Combine images into a single 5x5 image
combined_image = Image.new("RGB", (n * 32, n * 32))
for i in range(n**2):
    x_offset = i % n * 32
    y_offset = i // n * 32
    combined_image.paste(images[i], (x_offset, y_offset))

# Save the combined image to file
combined_image.save("combined_images.png")

# Add a new column to the DataFrame containing the RGB data for each image
df["image"] = df["structure"].apply(lambda x: visualize_structure(x))

# Remove the structure and structures columns from the DataFrame
df = df.drop(columns=["structure", "structures"])

# Change the is_stable column to a binary integer
df["stables"] = df["stables"].astype(int)

# Save the DataFrame to json file
df.to_json("training_data.json")
