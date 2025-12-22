from pymatgen.core import Composition
# from matminer.featurizers.composition import ElementFraction
from matminer.featurizers.composition import ElementProperty
import numpy as np
import random
from itertools import combinations
import pandas as pd

def magpie_it(compositions):
    Compositions = [Composition(c) for c in compositions]
    elementproperty = ElementProperty.from_preset("magpie")
    # print(elementproperty.feature_labels())
    # 4) Generate features:
    # features = feature_generator.featurize(entries)
    elementproperty_features = [elementproperty.featurize(C) for C in Compositions]
    return np.array(elementproperty_features)
    # print(elementproperty_features)
    # print(np.array(elementproperty_features).shape)

def fix_composition(vec):
    """
    Given a 1D array (5,) of floats in (0,1),
    normalize it and round it to two decimals so that:
      - sum == 1.000
      - all entries > 0
    Use the absolute value (TODO)
    Use softmax approach (TODO)
    If we can't fix it in a straightforward step, return None.
    """
    # Normalize to sum to 1 in floating-point
    vec = abs(vec)
    # vec = np.exp(vec)
    norm = vec / np.sum(vec)
    
    # Round each to three decimals
    rounded = np.round(norm, 3)
    # Compute difference from 1.00
    diff = 1.0 - np.sum(rounded)
    
    # If the sum is already good and all > 0, we are done.
    # But in practice, we often need to fix small differences:
    if abs(diff) < 1e-7 and np.all(rounded > 0):
        return rounded
    
    # Otherwise, try to fix small difference by adjusting
    # the largest component (or smallest if diff>0 and that
    # gives you trouble). Typically, the largest element is a safe bet:
    idx_largest = np.argmax(rounded)
    new_val = rounded[idx_largest] + diff
    
    # If new_val goes non-positive, return None (to signal failure)
    if new_val <= 0:
        return None
    
    # Update that entry
    rounded[idx_largest] = np.round(new_val, 3)
    
    # Final check
    if (abs(1.0 - np.sum(rounded)) < 1e-7) and np.all(rounded > 0):
        return rounded
    else:
        return None
    
def search_space_to_compositions(search_space):
    compositions = []
    for row in search_space:
        # Attempt to fix the composition. If it fails, regenerate row a few times
        comp = None
        retries = 0
        while comp is None and retries < 100:
            comp = fix_composition(row)
            if comp is None:
                # If we failed, generate a new row and retry
                row = np.random.randn(5)
            retries += 1
        
        # If we still failed after 100 attempts, 
        # raise an error or handle it in some other way
        if comp is None:
            raise ValueError("Could not generate a valid composition after 100 retries.")
        
        compositions.append(comp)
    
    return compositions

def combination_to_formula(elements, composition):
    """
    Convert a tuple of elements and a composition list
    into a single string formula, e.g.:
      ('Ti','V','Mn','Fe','Zn'), [0.293,0.052,0.322,0.141,0.192]
    --> 'Ti0.29V0.05Mn0.32Fe0.14Zn0.19'
    """
    # Build each "ElementFraction" piece
    parts = [f"{el}{val:.3f}" for el, val in zip(elements, composition)]
    # Join without spaces for a single formula string
    return "".join(parts)

def make_samples(selected_combinations, compositions):
    df_data = []
    # for combo in selected_combinations:
    #     for comp in compositions:
    #         formula = combination_to_formula(combo, comp)
    #         df_data.append({"Combination": combo, "Composition": comp.tolist(), "Formula": formula})
    for combo, comp in zip(selected_combinations, compositions):
        formula = combination_to_formula(combo, comp)
        df_data.append({"Combination": combo, "Composition": comp.tolist(), "Formula": formula})
    df = pd.DataFrame(df_data)
    samples = magpie_it(df["Formula"])    
    return samples, df

def make_combinations(random_init):
    elements = ["Sc", "Ti", "V", "Cr", "Mn", 
                "Fe", "Co", "Ni", "Cu", "Zn"]
    all_combinations = list(combinations(elements, 5))
    # selected_combinations = random.sample(all_combinations, random_init)
    repeated_combinations = all_combinations * (random_init // len(all_combinations) + 1)
    random.shuffle(repeated_combinations)
    # Select the first `random_init` samples
    selected_combinations = repeated_combinations[:random_init]
    return selected_combinations

def total_combinations():
    elements = ["Sc", "Ti", "V", "Cr", "Mn", 
                "Fe", "Co", "Ni", "Cu", "Zn"]
    all_combinations = list(combinations(elements, 5))
    # selected_combinations = random.sample(all_combinations, random_init)
    return all_combinations