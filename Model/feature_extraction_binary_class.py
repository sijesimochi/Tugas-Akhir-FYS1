import os
import pandas as pd
import numpy as np

# Define the main directory
main_dir = "C:/CJ/Coolyeah/CD/Coding/Tugas-Akhir-FYS1/dataset"

names = [
    "aldy",
    "fikri",
    "hayyul",
    "imam",
    "jauza",
    "lea",
    "pandu",
    "raja",
    "sije",
    "wawan",
]

# Initialize an empty list to store mean values
data = []

# Walk through the directories and process each CSV file
for subdir in ["bungkuk", "duduk", "jongkok", "jatoh"]:
    for sub_sub_dir_name in names:
        sub_sub_dir = os.path.join(main_dir, subdir, sub_sub_dir_name)

        # Determine the label based on the subdirectory
        label = str(1) if subdir == "jatoh" else str(0)

        # Check if the directory exists to avoid errors
        if os.path.exists(sub_sub_dir):
            # Process all csv files in the sub-sub-directory
            for file in os.listdir(sub_sub_dir):
                if file.endswith(".csv"):
                    file_path = os.path.join(sub_sub_dir, file)

                    # Read the CSV file
                    df = pd.read_csv(file_path)

                    # Drop the 'timestamp' column if it exists
                    if 'timestamp' in df.columns:
                        df = df.drop(columns=["timestamp"])

                    # Calculate the mean of each column
                    mean_values = df.mean().values  # Use df.var().values for variance, or df.mean().values for mean

                    # Append the mean values as a list to the data list, along with the label
                    data.append(np.append(mean_values, label))

# Convert the list of mean values to a DataFrame
# Extract column names from the first dataframe read and add 'label'
columns = df.columns.tolist() + ['label']
result_df = pd.DataFrame(data, columns=columns)

# Save the result DataFrame to a new CSV file
result_df.to_csv("transformed_var.csv", index=True)  # Change the filename appropriately for variance
