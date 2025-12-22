import pandas as pd
import random
import numpy as np
from itertools import combinations
from utility import search_space_to_compositions, make_combinations, make_samples, total_combinations

# All elements
elements = ["Sc", "Ti", "V", "Cr", "Mn", 
            "Fe", "Co", "Ni", "Cu", "Zn"]

# Step 1: Generate all combinations of length 5
element_combinations = list(combinations(elements, 5))

# Step 2: Randomly select 20 out of these
selected_combinations = random.sample(element_combinations, 20)

def generate_composition():
    """
    Generate a random composition of length 5, 
    all entries > 0, sum to 1, with two decimals.
    Retries until the criteria are met.
    """
    while True:
        # Generate 5 random integers in [1, 100]
        arr = np.random.randint(1, 101, size=5, dtype=int)
        # Normalize so that the sum is 1 before rounding
        arr = arr / arr.sum()
        # Round each value to 2 decimals
        arr_rounded = np.round(arr, 2)
        
        # Check criteria:
        #  1) sum of rounded entries must be 1.00
        #  2) all entries > 0
        # Use np.isclose(..., 1.0) if you want a small float tolerance
        if np.isclose(arr_rounded.sum(), 1.0) and np.all(arr_rounded > 0):
            return arr_rounded

# Step 3: Build the DataFrame
data = []
for combo in selected_combinations:
    composition = generate_composition()
    data.append({
        "Combination": combo,
        "Composition": composition.tolist()
    })

df = pd.DataFrame(data)
print(df)

# df = pd.read_csv('3d_metal_data00.csv')
# columns = ['composition','lat','sws0','B0','M_s','T_C','Pugh','Cauchy']
# df = df.copy().head(20)[columns]

# print(df)
# df.to_csv('3d_metal_data.csv', index=False)
