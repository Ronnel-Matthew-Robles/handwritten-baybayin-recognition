import os
import csv
import numpy as np

# Set the directory containing the CSV files
csv_dir = 'confusion_matrices'

# Get the list of CSV files in the directory
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

# Initialize an empty dictionary to store the confusion matrices
confusion_matrices = {}

# Iterate over the CSV files and store the confusion matrices in the dictionary
for csv_file in csv_files:
    with open(os.path.join(csv_dir, csv_file), 'r') as f:
        reader = csv.reader(f)
        matrix = np.array([list(map(int, row)) for row in reader])
        confusion_matrices[csv_file] = matrix

# Calculate the average confusion matrix
avg_matrix = np.mean(list(confusion_matrices.values()), axis=0)

# Write the average confusion matrix to a new CSV file
with open('average_confusion_matrix.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(avg_matrix)