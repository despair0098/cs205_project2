# Libraries
import pandas
import algs
import os

print("Welcome to Feature Selection Algorithm.")
file_path = input("Please enter file name: ")
while not os.path.exists(file_path):
    print("Error: file doesn't exist")
    file_path = input("Please enter file name: ")
file_path = pandas.read_csv(file_path, delim_whitespace=True, engine="python", header=None)

