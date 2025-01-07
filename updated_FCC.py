import pandas as pd
import os
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy
DATASET = "Food-101"
FOLDER_PATH = "/home/fahadk/Project/FCC/Results/Misc/NVP_Stacking/Images/"+DATASET

# Create a list to store DataFrames
dataframes = []


# Majority Vote
# # Ensure there are at least two or more files for majority voting
# if len(dataframes) < 2:
#     print("Not enough files for majority voting. Please provide at least two CSV files.")
# else:
#     # Combine Predicted_FCC from all files into one DataFrame
#     predicted_fcc = pd.concat([df["Predicted_FCC"] for df in dataframes], axis=1)
#     predicted_fcc.columns = [f"File{i+1}" for i in range(len(dataframes))]
#
#     # Apply majority voting
#     def majority_vote(row):
#         # Count occurrences of each prediction
#         vote_counts = Counter(row)
#         # Return the most common prediction
#         return vote_counts.most_common(1)[0][0]
#
#     predicted_fcc["MajorityVote"] = predicted_fcc.apply(majority_vote, axis=1)
#
#     # Merge majority vote back to the first file
#     final_file = dataframes[0].copy()
#     final_file["Final_Prediction"] = predicted_fcc["MajorityVote"]
#
#     # Filter rows where the MajorityVote matches the `Predicted_FCC`
#     selected_rows = final_file[final_file["Predicted_FCC"] == final_file["Final_Prediction"]]


def apply_weighted_voting(dataframes, weights, output_file="Metro_weighted_voting.csv"):
    """
    Applies weighted voting to combine predictions from multiple CSV files.

    Args:
        dataframes (list of pd.DataFrame): List of DataFrames containing `Predicted_FCC`.
        weights (list of int): List of weights corresponding to the reliability of each file.
        output_file (str): Path to save the resulting file with selected rows.

    Returns:
        pd.DataFrame: A DataFrame containing rows where the final weighted prediction matches the `Predicted_FCC`.
    """
    if len(dataframes) < 2:
        print("Not enough files for weighted voting. Please provide at least two CSV files.")
        return None

    if len(dataframes) != len(weights):
        raise ValueError("The number of weights must match the number of files.")

    # Combine `Predicted_FCC` from all files into one DataFrame
    predicted_fcc = pd.concat([df["Predicted_FCC"] for df in dataframes], axis=1)
    predicted_fcc.columns = [f"File{i+1}" for i in range(len(dataframes))]

    # Apply weighted voting
    def weighted_vote(row):
        # Create a dictionary to store the weighted vote counts
        vote_counts = Counter()

        # Add weighted votes for each prediction
        for i, value in enumerate(row):
            vote_counts[value] += weights[i]

        # Return the prediction with the highest weighted count
        return vote_counts.most_common(1)[0][0]

    predicted_fcc["WeightedVote"] = predicted_fcc.apply(weighted_vote, axis=1)

    # Merge weighted vote results back to the first file
    final_file = dataframes[0].copy()
    final_file["Final_Prediction"] = predicted_fcc["WeightedVote"]

    # Filter rows where the `WeightedVote` matches the `Predicted_FCC`
    # selected_rows = final_file[final_file["Predicted_FCC"] == final_file["Final_Prediction"]]

    # Save the selected rows
    final_file.to_csv(output_file, index=False)
    print(f"Selected rows based on weighted voting saved to '{output_file}'.")

    return final_file

# Example Usage
# Example file loading and weights
meta_targets = None
meta_features = []
# Read all CSV files from the folder
# for file_name in sorted(os.listdir(FOLDER_PATH)):
#     if file_name.endswith(".csv"):
#         file_path = os.path.join(FOLDER_PATH, file_name)
#         df = pd.read_csv(file_path)
#         dataframes.append(df)

for file_name in sorted(os.listdir(FOLDER_PATH)):
    if file_name.endswith(".csv"):
        file_path = os.path.join(FOLDER_PATH, file_name)
        df = pd.read_csv(file_path)
        if meta_targets is None:
            meta_targets = df["true_label"]  # Use y_true as target
        # Use probabilities or Max probability as meta-features
        features = df[["Predicted_FCC"]]
        meta_features.append(features)
        # dataframes.append(features)


# Concatenate meta-features from all base learners
X_train = pd.concat(meta_features, axis=1)
X_train = X_train.replace("omission", -1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_train, meta_targets, test_size=0.3, random_state=42
)

# Train the meta-learner
# meta_learner = LogisticRegression( random_state=42)
meta_learner = RandomForestClassifier(random_state=42)
meta_learner.fit(X_train, y_train)


# Evaluate the meta-learner
y_pred = meta_learner.predict(X_test)
y_proba = meta_learner.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Model Test Accuracy (Predicting Predicted_FCC): {accuracy * 100:.2f}%")
# Save predictions
predictions_df = pd.DataFrame({
    "true_label": y_test,
    "Max_Probability": numpy.max(y_proba,axis = 1),
    "predicted_label": y_pred,
})
predictions_df['Predicted_FCC'] = predictions_df.apply(
    lambda row: row['predicted_label'] if row['Max_Probability'] > 0.80 else 'omission',
    axis=1
)
predictions_df.to_csv(f"{DATASET}_stacking_{meta_learner.__class__.__name__}.csv", index=False)
# weights = [0.940, 0.943, 0.814, 0.941, 0.934 ]# Images Weights
# weights = [0.965,0.997,1.00,1.00,1.00] # Tabular Weights

# Apply weighted voting
# selected_rows = apply_weighted_voting(dataframes, weights)


# Recovery Blocks
# Ensure there are at least two files for Recovery Blocks
# if len(dataframes) < 2:
#     print("Not enough files for Recovery Blocks. Please provide at least two valid CSV files.")
# else:
# # Define the column and omission criteria
#     # Define the column and omission criteria
#     column_name = "Predicted_FCC"
#     omission_criteria = "omission"  # Replace with the exact value for omission
#
#     # Initialize the rows to process
#     rows_to_process = None
#     results = pd.DataFrame()
#
#     # Process each file in sequence
#     for i, df in enumerate(dataframes):
#         print(f"Processing file {i + 1}")
#
#         if rows_to_process is None:
#             # Start with all rows in the first file
#             valid_rows = df[df[column_name] != omission_criteria]
#             omission_rows = df[df[column_name] == omission_criteria]
#         else:
#             # Filter only the omitted rows from the previous file
#             current_rows = df.loc[rows_to_process.index]
#             valid_rows = current_rows[current_rows[column_name] != omission_criteria]
#             omission_rows = current_rows[current_rows[column_name] == omission_criteria]
#
#         # Append valid rows to the results
#         results = pd.concat([results, valid_rows])
#
#         # Update rows_to_process for the next iteration
#         rows_to_process = omission_rows
#
#     # Append remaining omissions
#     if not rows_to_process.empty:
#         print("Appending remaining omissions to the final results.")
#         results = pd.concat([results, rows_to_process])
#
#     # Save the final results
#     results.to_csv("NIDS_Recovery_Block.csv", index=False)
#     print("Final results (with omissions) saved to 'final_results_with_omissions.csv'.")