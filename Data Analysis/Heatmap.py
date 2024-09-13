import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
in_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Plate Absorbance Correction\PS BuOAC Correction Test Processed.csv"
out_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Plate Absorbance Correction"


def load_data(path_input):
    """Load CSV data into a Pandas DataFrame."""
    try:
        return pd.read_csv(path_input)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


# Main script execution
def main():
    # Load the data
    df = load_data(in_path)

    if df is not None:

        # Pivot the DataFrame to a format suitable for a heatmap
        heatmap_data = df.pivot(index='Well\nRow', columns='Well\nCol', values='282')
        print(heatmap_data)

        # Plot the heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='coolwarm', cbar=True)

        # Display the plot
        plt.title('Blank Corrected Absorbance of PS in BuOAc at 282 nm')
        plt.show()


if __name__ == "__main__":
    main()