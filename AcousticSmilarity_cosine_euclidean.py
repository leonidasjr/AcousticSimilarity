import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tkinter import filedialog
from tkinter import Tk

# Create a GUI window
root = Tk()
root.withdraw()  # Hide the main window

# Ask the user for the data frame file
file_path = filedialog.askopenfilename(title="Please select your data frame file")

# Load your data
df = pd.read_csv(file_path, delimiter='\t')

# Extract identifiers
identifiers = df.iloc[:, 0]

# Remove identifiers from data
df = df.iloc[:, 1:]

# Calculate the cosine similarity for each pair of voices
similarity_matrix_cosine = cosine_similarity(df)

# Calculate the Euclidean distance for each pair of voices
distance_matrix_euclidean = euclidean_distances(df)

# Round the similarity matrix to 2 decimal places
rounded_similarity_matrix_cosine = np.round(similarity_matrix_cosine, 2)

# Round the Euclidean distance matrix to 2 decimal places
rounded_distance_matrix_euclidean = np.round(distance_matrix_euclidean, 2)

# Transform Euclidean distances to 0-to-1 scale and multiply by 100
transformed_distance_matrix = 1 / (1 + distance_matrix_euclidean) * 100

# Ensure any value of 100.00 in the transformed matrix is set to 1.00
transformed_distance_matrix[transformed_distance_matrix == 100.00] = 1.00

# Ensure any value higher than 1 in the transformed matrix is set to a proportion between 0.91 and 0.99
transformed_distance_matrix[transformed_distance_matrix > 1] = 0.91 + (transformed_distance_matrix[transformed_distance_matrix > 1] - 1) * (0.99 - 0.91)

# again, checking values higher than 1
# If any value is still higher than 1 the transformed matrix, ensure this value to be set to a proportion between 0.95 and 0.99
transformed_distance_matrix[transformed_distance_matrix > 1] = 0.95 + (transformed_distance_matrix[transformed_distance_matrix > 1] - 1) * (0.99 - 0.95)

# Round the transformed distance matrix to 2 decimal places
rounded_transformed_distance_matrix = np.round(transformed_distance_matrix, 2)

# Convert the similarity and distance matrices into DataFrames
similarity_df_cosine = pd.DataFrame(rounded_similarity_matrix_cosine, index=identifiers, columns=identifiers)
distance_df_euclidean = pd.DataFrame(rounded_distance_matrix_euclidean, index=identifiers, columns=identifiers)
transformed_distance_df = pd.DataFrame(rounded_transformed_distance_matrix, index=identifiers, columns=identifiers)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("Cosine Similarity Matrix:")
print(similarity_df_cosine)

print("\nEuclidean Distance Matrix:")
print(distance_df_euclidean)

print("\nTransformed Euclidean Distance Matrix:")
print(transformed_distance_df)

print("")
print('Pairwise Acoustic similarity')
print("")

# Perform pairwise comparisons
for i in range(len(similarity_df_cosine)):
    for j in range(i+1, len(similarity_df_cosine)):
        print(f'Acoustic similarity (cosine) ==> {similarity_df_cosine.index[i]} & {similarity_df_cosine.index[j]} = {similarity_df_cosine.iloc[i, j]}')
        print(f'Acoustic distance (euclidean) ==> {distance_df_euclidean.index[i]} & {distance_df_euclidean.index[j]} = {distance_df_euclidean.iloc[i, j]}')
        print(f'Transformed acoustic distance ==> {transformed_distance_df.index[i]} & {transformed_distance_df.index[j]} = {transformed_distance_df.iloc[i, j]}')
        print('')

# Write the dataframes to tab-delimited txt files
similarity_df_cosine.to_csv(file_path.replace('.txt', '_similarity_matrix_cosine.txt'), sep='\t')
distance_df_euclidean.to_csv(file_path.replace('.txt', '_distance_matrix_euclidean.txt'), sep='\t')
transformed_distance_df.to_csv(file_path.replace('.txt', '_transformed_distance_matrix.txt'), sep='\t')
