import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define file paths
in_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\12-Sep-2024\PS Sty Mixtures 12-Sep Blank and Background Corrected.csv"
out_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\12-Sep-2024"


def load_data(path_input):
    """Load CSV data into a Pandas DataFrame."""
    try:
        return pd.read_csv(path_input)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def group_and_calculate(df, operation='mean', group_size=4):
    """Group numeric data by a defined number of rows and calculate the specified operation."""
    numeric_df = df.select_dtypes(include='number')

    if operation == 'mean':
        grouped_df = numeric_df.groupby(numeric_df.index // group_size).mean()
    elif operation == 'std':
        grouped_df = numeric_df.groupby(numeric_df.index // group_size).std()

    # Reset index
    grouped_df = grouped_df.reset_index(drop=True)

    # Merge with non-numeric data
    non_numeric_df = df.select_dtypes(exclude='number')
    non_numeric_df = non_numeric_df.groupby(non_numeric_df.index // group_size).first().reset_index(drop=True)
    return pd.concat([non_numeric_df, grouped_df], axis=1)


def save_dataframe(df, filename, output_dir):
    """Save the DataFrame to a CSV file."""
    try:
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    except Exception as e:
        print(f"Error saving file: {e}")


def filter_by_rsd(df, column, threshold):
    """Remove rows where RSD exceeds the given threshold for a specific column."""
    try:
        indexes_to_remove = df[df[column] > threshold].index
        return df.drop(indexes_to_remove)
    except KeyError:
        print(f"Column {column} not found.")
        return df


def plot_absorbance(df, x_col_start, x_col_end, title="Absorbance Spectra", num_samples=4, wavelength_range=(220, 1000), ylim=(-1.0, 2)):
    """Plot absorbance spectra for the selected number of samples."""
    x = [int(i) for i in df.columns[x_col_start:x_col_end].values]
    y = [df.iloc[i, x_col_start:x_col_end].values for i in range(num_samples)]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(len(y)):
        ax.plot(x, y[i], label=f'{df.iloc[i, 1]}')

    ax.set_xlim(wavelength_range)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(wavelength_range[0], wavelength_range[1], 10))
    ax.grid(True, 'major', 'y')
    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (AU)")
    ax.legend()
    plt.show()


# Main script execution
def main():
    # Load the data
    df = load_data(in_path)

    if df is not None:
        # Calculate averages and standard deviations
        averaged_df = group_and_calculate(df, operation='mean', group_size=2)
        std_df = group_and_calculate(df, operation='std', group_size=2)
        rsd=std_df.select_dtypes(include='number')/averaged_df.select_dtypes(include='number')*100

        # Save the results
        save_dataframe(averaged_df, 'PS Sty Mixtures 12-Sep Blank and Background Corrected Averaged.csv', out_path)
        save_dataframe(std_df, 'PS Sty Mixtures 12-Sep Blank and Background Corrected Std Dev.csv', out_path)
        save_dataframe(rsd, 'PS Sty Mixtures 12-Sep Blank and Background Corrected RSD.csv', out_path)

        # Filter and remove data based on RSD and sort the dataframe by desired column
        # filtered_df = filter_by_rsd(averaged_df, column='260', threshold=100)
        sorted_df = averaged_df.sort_values(by=['260'], ascending=False)

        # Plot the absorbance spectra
        plot_absorbance(sorted_df,
                        x_col_start=3,
                        x_col_end=len(sorted_df.columns),
                        title="Absorbance Spectra for PS in BuOAc at Varying Concentrations",
                        num_samples=10,
                        wavelength_range=(220, 400),
                        ylim=(-1,5))


if __name__ == "__main__":
    main()
