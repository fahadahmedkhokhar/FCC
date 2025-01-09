import pandas as pd
import sklearn
import numpy
import csv
import os
def compute_omission_metrics(y_true, y_wrapper, y_clf, reject_tag='omission'):
    """
    Assumes that y_clf may have omissions, labeled as 'reject_tag'
    :param y_true: the ground truth labels
    :param y_wrapper: the prediction of the SPROUT (wrapper) classifier
    :param y_clf: the prediction of the regular classifier
    :param reject_tag: the tag used to label rejections, default is None
    :return: a dictionary of metrics
    """
    met_dict = {}
    met_dict['alpha'] = sklearn.metrics.accuracy_score(y_true, y_clf)
    met_dict['eps'] = 1 - met_dict['alpha']
    met_dict['phi'] = numpy.count_nonzero(y_wrapper == reject_tag) / len(y_true)
    met_dict['alpha_w'] = sum(y_true == y_wrapper) / len(y_true)
    met_dict['eps_w'] = 1 - met_dict['alpha_w'] - met_dict['phi']
    met_dict['phi_c'] = sum(numpy.where((y_wrapper == reject_tag) & (y_clf == y_true), 1, 0))/len(y_true)
    met_dict['phi_m'] = sum(numpy.where((y_wrapper == reject_tag) & (y_clf != y_true), 1, 0)) / len(y_true)
    met_dict['eps_gain'] = 0 if met_dict['eps'] == 0 else (met_dict['eps'] - met_dict['eps_w']) / met_dict['eps']
    met_dict['phi_m_ratio'] = 0 if met_dict['phi'] == 0 else met_dict['phi_m'] / met_dict['phi']
    # met_dict['overall'] = 2*met_dict['eps_gain']*met_dict['phi_m_ratio']/(met_dict['eps_gain'] + met_dict['phi_m_ratio'])
    return met_dict

def save_or_append_dict_plain(data, file_path):
    # Convert the dictionary to a plain text format
    # data_to_write = str(dictionary)
    headers = data[0].keys()


    # Check if the file exists
    if os.path.exists(file_path):
        # Append the text if the file exists
        with open(file_path, 'a') as file:
            writer = csv.DictWriter(file, fieldnames=headers, delimiter='\t')  # Using tab as a delimiter
            # writer.writeheader()  # Write the column names
            writer.writerows(data)  # Write the rows
            # file.write("\n" + da)  # Add a newline before appending
        print(f"Appended to {file_path}")
    else:
        # Create the file and write the text if it doesn't exist
        with open(file_path, 'w') as file:
            writer = csv.DictWriter(file, fieldnames=headers, delimiter='\t')  # Using tab as a delimiter
            writer.writeheader()  # Write the column names
            writer.writerows(data)  # Write the rows
        print(f"Created and wrote to {file_path}")

def read_all_csv_files(folder_path):
    """
    Reads all CSV files in the specified folder and returns a list of DataFrames.

    Args:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        list: A list of DataFrames, one for each CSV file.
        list: A list of filenames corresponding to the DataFrames.
    """
    # List to store DataFrames and filenames
    dataframes = []
    filenames = []

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                dataframes.append(df)
                filenames.append(file_name)
                print(f"Loaded: {file_name}")
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    return dataframes, filenames

output_file = "results.txt"
metrices =[]
# file_names = None
# Example Usage
folder_path = "/home/fahadk/Project/FCC/Results/Final_Results/Tabular_wu/NIDS"#Results/Final_Results/Tabular/Metro"  # Replace with the path to your folder
# dataframes, filenames = read_all_csv_files(folder_path)
for df,filenames in zip(*read_all_csv_files(folder_path)):

    y_true = df['true_label'].to_numpy()

    y_clf = df['predicted_label'].to_numpy()

    y_wrapper = df['Predicted_FCC'].apply(lambda x: int(float(x)) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x)
    # y_wrapper = df['Predicted_FCC'].apply(
    #     lambda x: int(x) if str(x).isdigit() else "omission"
    # )
    # y_wrapper = y_wrapper.apply(lambda x: int(x) if x in [0.0, 1.0] else x)

    # y_wrapper = y_wrapper.to_numpy()

    # y_wrapper = numpy.where(y_wrapper == None, df['Final_Prediction'], None)
    # y_wrapper = df['Final_Prediction']

    metrics = compute_omission_metrics(y_true,y_wrapper, y_clf)
    metrics['file_name'] = filenames
    metrices.append(metrics)
save_or_append_dict_plain(metrices, output_file)
