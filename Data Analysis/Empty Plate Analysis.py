import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define file paths
in_path = [
    r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Empty Plate Specs\Processed\Empty Plate 1a.csv",
    r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Empty Plate Specs\Processed\Empty Plate 1b.csv",
    r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Empty Plate Specs\Processed\Empty Plate 1c.csv",
    r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Empty Plate Specs\Processed\Empty Plate 2a.csv",
    r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Empty Plate Specs\Processed\Empty Plate 2b.csv",
    r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Empty Plate Specs\Processed\Empty Plate 2c.csv",
    r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Empty Plate Specs\Processed\Empty Plate 3a.csv",
    r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Empty Plate Specs\Processed\Empty Plate 3b.csv",
    r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Empty Plate Specs\Processed\Empty Plate 3c.csv"
]
out_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Empty Plate Specs"


def load_data(path_input):
    """Load CSV data into a Pandas DataFrame."""
    try:
        return pd.read_csv(path_input)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def save_dataframe(df, filename, output_dir):
    """Save the DataFrame to a CSV file."""
    try:
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    except Exception as e:
        print(f"Error saving file: {e}")


def calculate_average_and_std(df_list):
    """Calculate element-wise average and standard deviation for numeric data."""
    # Extract numeric columns
    numeric_cols = df_list[0].select_dtypes(include=[np.number]).columns

    # Stack only the numeric data from each DataFrame along a new axis
    numeric_data = np.stack([df[numeric_cols].to_numpy() for df in df_list])

    # Element-wise mean and standard deviation
    avg = np.mean(numeric_data, axis=0)
    std_dev = np.std(numeric_data, axis=0)

    # Create DataFrames for the results
    averaged_df = pd.DataFrame(avg, index=df_list[0].index, columns=numeric_cols)
    std_dev_df = pd.DataFrame(std_dev, index=df_list[0].index, columns=numeric_cols)

    return averaged_df, std_dev_df


# Main script execution
def main():
    # Load the data
    df_list = [load_data(i) for i in in_path]

    # Calculate averages and standard deviations for numeric data
    averaged_df, std_dev_df = calculate_average_and_std(df_list)
    print(averaged_df)

    # Relative standard deviation (RSD) for numeric columns only
    rsd_df = pd.DataFrame()
    for col in averaged_df.columns:
        rsd_df[col] = (std_dev_df[col] / averaged_df[col]) * 100

    # Optionally, merge non-numeric data back
    non_numeric_cols = df_list[0].select_dtypes(exclude=[np.number]).columns
    if not non_numeric_cols.empty:
        non_numeric_data = df_list[0][non_numeric_cols]
        averaged_df = pd.concat([non_numeric_data, averaged_df], axis=1)
        std_dev_df = pd.concat([non_numeric_data, std_dev_df], axis=1)
        rsd_df = pd.concat([non_numeric_data, rsd_df], axis=1)

    # Print the results
    print("Averaged Data:")
    print(averaged_df)
    print("\nStandard Deviation Data:")
    print(std_dev_df)
    print("\nRelative Standard Deviation (RSD):")
    print(rsd_df)

    # Save the results
    save_dataframe(averaged_df, 'average.csv', out_path)
    save_dataframe(std_dev_df, 'std dev.csv', out_path)
    save_dataframe(rsd_df, 'RSD.csv', out_path)


if __name__ == "__main__":
    main()
