import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Define file paths
raw_data = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\12-Sep-2024\PS Sty Mixtures 12 Sept RAW processed.csv"
plate_background = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Empty Plate Specs\Processed\Empty Plate 1c.csv"
concs = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\concentrations.csv"
output = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\12-Sep-2024"


def load_data(path_input):
    """Load CSV data into a Pandas DataFrame."""
    try:
        return pd.read_csv(path_input)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def separate_columns(df):
    """Separate numeric and non-numeric columns, while keeping 'Well\nCol' aside."""
    numeric_cols = df.select_dtypes(include='number').drop(columns=['Well\nCol'])
    non_numeric_cols = df.select_dtypes(exclude='number')
    col_column = df['Well\nCol']
    # Capture original column order
    original_columns = df.columns.tolist()
    return numeric_cols, non_numeric_cols, col_column, original_columns


def subtract_background(numeric_raw, numeric_plate):
    """Subtract the plate background from the raw data."""
    return numeric_raw.subtract(numeric_plate, fill_value=0)


def subtract_blank_row(df):
    """Subtract the first row (blank) from the entire DataFrame."""
    blank_row = df.iloc[[0]].values[0]
    return df.apply(lambda row: row - blank_row, axis=1)


def recombine_data(numeric_data, non_numeric_data, col_column, original_columns):
    """Recombine numeric, 'Well\nCol', and non-numeric columns in the original order."""
    combined_df = pd.concat([col_column, non_numeric_data, numeric_data], axis=1)

    # Reorder columns based on original column order
    combined_df = combined_df.reindex(columns=original_columns)
    return combined_df


def group_and_calculate(df, operation='mean', group_size=4):
    """Group numeric data by a defined number of rows and calculate the specified operation. Should only be used on
    separated dataframe (i.e., numeric columns only)."""

    if operation == 'mean':
        grouped_df = df.groupby(df.index // group_size).mean()
    elif operation == 'std':
        grouped_df = df.groupby(df.index // group_size).std()

    # Reset index
    grouped_df = grouped_df.reset_index(drop=True)

    return grouped_df


def combine_sample_names(df, sample_names, blanks=None):
    if blanks is None:
        joined = pd.concat([sample_names, df.reset_index(drop=True)], axis=1)
        return joined
    elif type(blanks) is int:
        joined = pd.concat([sample_names, df.drop([i for i in range(blanks)]).reset_index(drop=True)], axis=1)
        return joined
    else:
        return


def separate_subtract_and_recombine(raw_df, plate_data):
    """Separates numeric from non-numeric columns, subtracts plate background and blank background,
    then recombines all columns to original order."""

    # Separate numeric and non-numeric columns
    numeric_cols_raw, non_numeric_cols_raw, col_column_raw, original_columns_raw = separate_columns(raw_df)
    numeric_cols_plate, non_numeric_cols_plate, _, _ = separate_columns(plate_data)

    # Subtract plate background and blank row
    plate_background_removed = subtract_background(numeric_cols_raw, numeric_cols_plate)
    blank_removed = subtract_blank_row(plate_background_removed)

    # Recombine columns, respecting original order
    final_plate = recombine_data(blank_removed, non_numeric_cols_raw, col_column_raw, original_columns_raw)

    return final_plate


def save_dataframe(df, filename, output_dir):
    """Save the DataFrame to a CSV file."""
    try:
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    except Exception as e:
        print(f"Error saving file: {e}")


def plot_heatmap(df, value_col, title, ax):
    """Plot a heatmap from the DataFrame."""
    heatmap_data = df.pivot(index='Well\nRow', columns='Well\nCol', values=value_col)
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='coolwarm', cbar=True, ax=ax)
    ax.set_title(title)


def plot_line(df, x_col_start, x_col_end, ax, title="Absorbance Spectra", num_samples=4, wavelength_range=(220, 1000),
              ylim=(-1.0, 2)):
    """Plot absorbance spectra for the selected number of samples."""
    x = [int(i) for i in df.columns[x_col_start:x_col_end].values]
    y = [df.iloc[i, x_col_start:x_col_end].values for i in range(num_samples)]

    for i in range(len(y)):
        ax.plot(x, y[i], label=f'{df.iloc[i, 1]}')

    ax.set_xlim(wavelength_range)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(wavelength_range[0], wavelength_range[1], 10))
    ax.grid(True, 'major', 'y')
    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (AU)")


def main():
    # Load the data
    raw_df = load_data(raw_data)
    plate_df = load_data(plate_background)
    concs_df = load_data(concs)

    raw_numeric, _, _, _ = separate_columns(raw_df)
    plate_numeric, _, _, _ = separate_columns(plate_df)
    pate_removed = subtract_background(raw_numeric, plate_numeric)
    processed = subtract_blank_row(pate_removed)

    averaged = group_and_calculate(processed, 'mean', 2)
    joined = combine_sample_names(averaged, concs_df, 2)
    print(joined)

    # Process data for plotting
    final_plate = separate_subtract_and_recombine(raw_df, plate_df)

    # Set up subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))

    # Plot heatmap on row 0 col 1 subplot
    plot_heatmap(final_plate,
                 value_col='260',
                 title='Plate & Blank Corrected Absorbance of PS/Styrene Mixtures in BuOAc at 260 nm',
                 ax=axes[0, 1])

    # Plot second heatmap on row 1 col 1 subplot
    plot_heatmap(final_plate,
                 value_col='282',
                 title='Plate & Blank Corrected Absorbance of PS/Styrene Mixtures in BuOAc at 282 nm',
                 ax=axes[1, 1])

    # Plot absorbance spectra for plate + blank background corrected samples on row 0 col 0 subplot
    plot_line(final_plate,
              x_col_start=3,
              x_col_end=final_plate.shape[1],
              ax=axes[0, 0],
              title="Plate & Blank Corrected Absorbance Spectra of PS/Styrene Mixtures in BuOAc",
              num_samples=final_plate.shape[0],
              wavelength_range=(220, 400),
              ylim=(-1, 2))

    # Adjust layout
    plt.tight_layout()
    # plt.show()

    # save_dataframe(final_plate, "PS Sty Mixtures 12-Sep Blank and Background Corrected.csv", output)


if __name__ == "__main__":
    main()
