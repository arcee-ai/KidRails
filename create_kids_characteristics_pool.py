import json
import random
import pandas as pd

import csv
import json


# Load the JSON file
file_path = './kids_characteristics_per_developmental_stage.json'
pool_size = 1000
with open(file_path, 'r') as file:
    data = json.load(file)

# Function to create a pool of kids' characteristics
def create_kids_characteristics_pool(data, pool_size):
    kids_pool = []
    
    for _ in range(pool_size):
        kid = {}
        
        # Randomly select a developmental stage
        stage = random.choice(data["developmental_stages"])
        # Randomly select an age within the age range
        kid["age"] = random.choice(stage["age_range"])
        kid["characteristics"] = stage["characteristics"]
        # Randomly select personality traits
        personality_traits = random.choice(stage["personality_traits"])
        adjective_type = random.choice(["positive_adjectives", "negative_adjectives"])
        if adjective_type == "positive_adjectives":
          kid["personality_trait"] = ""
          kid["adjectives"] = random.sample(personality_traits["positive_adjectives"], 2)
        else:
          kid["personality_trait"] = personality_traits["item"]
          kid["adjectives"] = random.sample(personality_traits["negative_adjectives"], 2)
        

        # Randomly select two fields of interest
        kid["interest"] = random.sample(stage["interests"], 2)
        
        kids_pool.append(kid)
    
    return kids_pool

# Create the pool
kids_characteristics_pool = create_kids_characteristics_pool(data, pool_size)

# Convert the pool to a DataFrame
kids_df = pd.DataFrame(kids_characteristics_pool)


# Optional: Save the DataFrame to a CSV file
output_csv_path = 'kids_characteristics_pool.csv'
kids_df.to_csv(output_csv_path, index=False)

# Print a sample of the pool
print(kids_characteristics_pool[:5])

json_data = kids_df.to_json(orient='records', indent=4)


# Write the JSON data to a file
with open('./kids_characteristics_pool.json', 'w') as file:
    file.write(json_data)


