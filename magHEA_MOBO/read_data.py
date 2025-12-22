import pickle

# Specify the path to your pickle file
i=10
file_path = f'model_weights/{i}/data.pickle'

# Open and load the pickle file
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Access the stored data
X_train, y_train = data['train']
X_candidate, y_candidate = data['test']

# Print or use the data
#print("X_train:", X_train)
print("y_train:", y_train)
#print("X_candidate:", X_candidate)
#print("y_candidate:", y_candidate)
