import pandas as pd

df = pd.read_csv("3d_metal_data.csv")

# Define the number of rows per sub-dataframe
chunk_size = 10

# Split the DataFrame into chunks of 10 rows
sub_dfs = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
target_props = ['B0', 'mag']
# Calculate average and variance for each chunk
results = []
for idx, sub_df in enumerate(sub_dfs):
    avg = sub_df[target_props].mean()  # Calculate mean
    var = sub_df[target_props].var()   # Calculate variance
    results.append({'Chunk': idx + 1, 'Average': avg, 'Variance': var})

# Convert results to a DataFrame for better presentation
result_df = pd.DataFrame({
    'Chunk': [res['Chunk'] for res in results],
    'Average': [res['Average'].to_dict() for res in results],
    'Variance': [res['Variance'].to_dict() for res in results]
})

print(result_df)