# Libraries
import pandas
import os
import algs

print("Welcome to Feature Selection Algorithm.")

# Input and validate file path
file_path = input("Please enter file name: ")
while not os.path.exists(file_path):
    print("Error: file doesn't exist")
    file_path = input("Please enter file name: ")

# Load and normalize data
df = pandas.read_csv(file_path, delim_whitespace=True, engine="python", header=None)
# this reorders the column. The class label is at the last column which is column 8. 
df_reordered = df.iloc[:, [8, 1, 2, 3, 4, 5, 6, 7]]
# Creates a list of features that is inside the dataset
class_features = df[8].unique()
print(f"The features inside the data: {class_features}")

# Converts the features to be integers
df_reordered['class'] = pandas.factorize(df[8])[0]
# Prints the features that are converted to integers.
print(f"Here is the converted corresponding features: {df_reordered['class'].unique()}")
df_reordered = df_reordered.iloc[:, [8, 1, 2, 3, 4, 5, 6, 7]]

for c in df_reordered.columns:
    df_reordered[c] = (df_reordered[c] - df_reordered[c].mean()) / df_reordered[c].std()

# Convert dataframe to list
data = df_reordered.values.tolist()

# Automatically use all feature indices (excluding class label at index 0)
listOfFeatures = list(range(1, len(df_reordered.columns)))  # Start from 1 since column 0 is class label
numInstances = len(df_reordered)
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


"""
https://stackoverflow.com/questions/55852570/how-can-i-get-all-the-unique-categories-within-my-dataframe-using-python
https://www.geeksforgeeks.org/change-the-order-of-a-pandas-dataframe-columns-in-python/
"""