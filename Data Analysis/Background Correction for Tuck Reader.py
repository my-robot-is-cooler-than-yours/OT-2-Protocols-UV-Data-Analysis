import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math
from cycler import cycler
from scipy.optimize import minimize

mpl.rcParams.update({'font.size': 12})
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['font.family'] = 'Times New Roman'

# Define  custom color palette
custom_colors = [
    '#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
    '#DDCC77', '#CC6677', '#882255', '#AA4499',  # Original palette
    '#661100', '#6699CC', '#AA4466', '#4477AA', '#228833',
    '#66CCEE', '#EEDD88', '#EE6677', '#AA3377', '#BBBBBB',
    '#333333', '#FFDD44', '#9988CC', '#66AA77', '#117755'
]

# Set the color cycle using plt.rc
plt.rc('axes', prop_cycle=cycler('color', custom_colors))

# Define file paths
unprocessed = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\12-Sep-2024\PS Sty Mixtures 12 Sept RAW.csv"
raw_data = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\12-Sep-2024\PS Sty Mixtures 12 Sept RAW processed.csv"
plate_background = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Empty Plate Specs\Processed\Empty Plate 1c.csv"
concs = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\concentrations.csv"
output = ""


def load_data(path_input):
    """Load CSV data into a Pandas DataFrame."""
    try:
        return pd.read_csv(path_input)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def load_data_new(path):
    """
    Designed for new data output with the PRD plate reader.
    Loads CSV with no headers then appends wavelength values.
    """

    # Load CSV to df with no headrs
    df = pd.read_csv(path, header=None)

    # Rename headers to wavelengths from 220 to 1000
    df.columns = ['Row/Col'] + list(range(220, 1001))

    # Sort the dataframe
    df_sorted = df.sort_values(by=['Row/Col']).reset_index(drop=True)

    return df_sorted


def reformat_df(df, meta_rows):
    # Remove the metadata rows
    df = df.iloc[meta_rows:].reset_index(drop=True)

    # Preserve the first two column headers ('Well Row' and 'Well Col')
    new_columns = ['Well\nRow', 'Well\nCol'] + df.iloc[0, 2:].tolist()

    # Set the new column headers
    df.columns = new_columns

    # Remove the row that was used for new column headers
    df = df.drop(0).reset_index(drop=True)

    # Convert col column to int
    df['Well\nCol'] = pd.to_numeric(df['Well\nCol'], errors='coerce')

    # Sort by column values to ensure wells are read top to bottom across the plate
    df_sorted = df.sort_values(by=['Well\nCol', 'Well\nRow'], ascending=[True, True]).reset_index(drop=True)

    return df_sorted


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
    """Combines a dataframe containing sample names with another dataframe.
    Option to specify how many blanks were taken to avoid misalignment. """
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


def plot_line(df, x_col_start, x_col_end, ax, title="Absorbance Spectra", samples_start=0, samples_end=4,
              wavelength_range=(220, 1000),
              ylim=(-1.0, 2)):
    """Plot absorbance spectra for the selected number of samples."""
    x = [int(i) for i in df.columns[x_col_start:x_col_end].values]  # Wavelength values
    y = [df.iloc[i, x_col_start:x_col_end].values for i in range(samples_start, samples_end)]  # Absorbance values

    for i in range(samples_start, samples_end):
        ax.plot(x, df.iloc[i, x_col_start:x_col_end].values, label=f'{df.iloc[i, 1]}')  # Use the index as the label

    ax.set_xlim(wavelength_range)
    ax.set_ylim(ylim)

    ax.set_xticks(np.arange(wavelength_range[0], wavelength_range[1], 10))

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)

    # Change spine width
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', pad=15)

    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (AU)")

    ax.grid(True, linestyle='-', linewidth=0.2, which='major', axis='both')
    ax.legend(loc='best', fontsize=8)


def main():
    raw_styrene = load_data(
        r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Styr 0.0250 mgmL Cuvette Processed.csv")
    raw_polystyrene = load_data(
        r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\PS 0.250 mgmL Cuvette Processed.csv")

    raw_df = load_data(raw_data)
    plate_df = load_data(plate_background)
    concs_df = load_data(concs)
    unprocessed_df = load_data(unprocessed)

    # Process data for plotting
    final_plate = separate_subtract_and_recombine(raw_df, plate_df)

    num_styrene, non_numeric_cols, col_column, original_columns = separate_columns(raw_styrene)
    num_polystyrene, _, _, _ = separate_columns(raw_polystyrene)

    # Plot

def main_2(): # Linear fitting code

    styrene_spectrum = load_data(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Styrene & PS Cuvette Specs\Styr 0.00625 mgml Processed.csv")
    polystyrene_spectrum = load_data(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Styrene & PS Cuvette Specs\PS 0.1042 mgml Processed.csv")

    concs_df = load_data(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\Concentrations_Duplicated.csv")

    raw_df = load_data(raw_data)
    plate_df = load_data(plate_background)

    # Process data for plotting
    final_plate = separate_subtract_and_recombine(raw_df, plate_df)

    num_styrene, non_numeric_cols, col_column, original_columns = separate_columns(styrene_spectrum)
    num_polystyrene, _, _, _ = separate_columns(polystyrene_spectrum)

    # Convert the Dataframes to NumPy arrays for easier calculations
    styrene_spectrum = num_styrene.values[0]  # 1D array for styrene absorbance
    polystyrene_spectrum = num_polystyrene.values[0]  # 1D array for polystyrene absorbance

    # Slice the data to restrict range to where most changes occur
    styrene_spectrum = styrene_spectrum[40:101]
    polystyrene_spectrum = polystyrene_spectrum[40:101]

    # Initialize x and y lists to store ratio_actual and ratio_pred
    x = []
    y = []

    for i in range(4, final_plate.shape[0]): # Start at index 4 so blanks not included
        unknown = final_plate.iloc[i, 3:].values  # 1D array for the unknown mixture absorbance at index i (represents each well read vertically)
        unknown_spectrum = unknown[40:101] # Slice to appropriate range

        # Define the objective function that calculates the error (residual) between the mixture and the linear combination of styrene and polystyrene
        def residuals(coeffs):
            c_styrene, c_polystyrene = coeffs
            combined_spectrum = c_styrene * styrene_spectrum + c_polystyrene * polystyrene_spectrum
            return np.sum((unknown_spectrum - combined_spectrum) ** 2)  # Sum of squared residuals

        # Initial guess for the concentrations
        initial_guess = [0.5, 0.5]

        # Set bounds to ensure the coefficients are non-negative
        bounds = [(0, None), (0, None)]  # (min, max) for each coefficient

        # Minimize the residuals to find the best coefficients
        result = minimize(residuals, initial_guess, bounds=bounds)

        # Extract the fitted coefficients (concentrations)
        c_styrene_opt, c_polystyrene_opt = result.x

        # Calculate ratio for predicted component coefficients and actual concentrations
        ratio_pred = (c_styrene_opt / c_polystyrene_opt)
        ratio_actual = (concs_df.iloc[i, 0] / concs_df.iloc[i, 1])

        # Append the values to x and y
        x.append(ratio_actual)
        y.append(ratio_pred)

        # Calculate the fitted spectrum
        fitted_spectrum = c_styrene_opt * styrene_spectrum + c_polystyrene_opt * polystyrene_spectrum

        # Calculate R^2
        # SS_res: Sum of squared residuals
        SS_res = np.sum((unknown_spectrum - fitted_spectrum) ** 2)

        # SS_tot: Total sum of squares (variance of the observed mixture)
        SS_tot = np.sum((unknown_spectrum - np.mean(unknown_spectrum)) ** 2)

        # Get R^2
        R_squared = 1 - (SS_res / SS_tot)

        # print(f"Coefficient of styrene: {c_styrene_opt: .4f}")
        # print(f"Coefficient of polystyrene: {c_polystyrene_opt: .4f}")
        # print(f"Ratio of c(Styr) to c(p[Styr]): {ratio_pred: .4f}, {ratio_actual: .4f}")
        # print(f"R Squared Value: {R_squared: .2f}")

    # Convert x and y to NumPy arrays
    x = np.array(x)
    y = np.array(y)

    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 5))

    plt.scatter(x, y, color='black', label='Predicted vs Actual Ratio', s=25)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)

    # Change spine width
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', pad=10)

    ax.set_title("Predicted vs Actual Ratio of Styrene/Polystyrene")
    ax.set_xlabel("Actual Ratio")
    ax.set_ylabel("Predicted Ratio")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-0.25)

    ax.grid(True, linestyle='-', linewidth=0.2, which='major', axis='both')
    ax.legend(loc='best', fontsize=8)

    plt.savefig(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\model linear fit.png")

    # wavelengths = superimp.columns.astype(float)[40:101]
    #
    # plt.plot(wavelengths, unknown_spectrum, label='Observed Mixture Spectrum', color='black')
    # plt.plot(wavelengths, fitted_spectrum, label='Fitted Spectrum', linestyle='--', color='blue')
    # plt.plot(wavelengths, styrene_spectrum, label='Styrene Component', linestyle='-.', color='red')
    # plt.plot(wavelengths, polystyrene_spectrum, label='Polystyrene Component', linestyle='-.',
    #          color='green')
    #
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Absorbance')
    # plt.legend()
    # plt.savefig(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Styrene & PS Cuvette Specs\diluted test.png")

def main_3():
    path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\240918_1620.csv"

    new_raw_data = load_data(path)

    test = load_data_new(path)

    print(test)

    # # Set up subplots
    fig, axes = plt.subplots(figsize=(8, 5))

    plot_line(test,
              x_col_start=1,
              x_col_end=test.shape[1],
              ax=axes,
              title="Test",
              samples_start=0,
              samples_end=1,
              wavelength_range=(220, 1000),
              ylim=(-1, 3.5)
              )
    # plt.show()


if __name__ == "__main__":
    main_3()
