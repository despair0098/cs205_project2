# Libraries
import pandas
import algs
import os

print("Welcome to Feature Selection Algorithm.")

# Input and validate file path
file_path = input("Please enter file name: ")
while not os.path.exists(file_path):
    print("Error: file doesn't exist")
    file_path = input("Please enter file name: ")

# Load and normalize data
df = pandas.read_csv(file_path, delim_whitespace=True, engine="python", header=None)
for c in df.columns:
    df[c] = (df[c] - df[c].mean()) / df[c].std()

# Convert dataframe to list
data = df.values.tolist()

# Automatically use all feature indices (excluding class label at index 0)
listOfFeatures = list(range(1, len(df.columns)))  # Start from 1 since column 0 is class label
numInstances = len(df)
print(f"Detected {len(listOfFeatures)} features, with {numInstances} instances")

# Choose algorithm
print("\nType the number of the algorithm you want to run:")
print("1. Forward Selection")
print("2. Backward Elimination")

choice2 = int(input("Your choice: "))
if choice2 == 1:
    print("Running Forward Selection...")
    algs.forwardSelection(data, listOfFeatures)
elif choice2 == 2:
    print("Running Backward Elimination...")
    algs.backwardElimination(data, listOfFeatures)
else:
    print("Invalid option.")
    exit(0)
