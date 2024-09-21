import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from cycler import cycler
from scipy.optimize import minimize
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

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

    return df


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
    """Separate numeric and non-numeric columns."""
    numeric_cols = df.select_dtypes(include='number')
    non_numeric_cols = df.select_dtypes(exclude='number')

    # Capture original column order
    original_columns = df.columns.tolist()
    return numeric_cols, non_numeric_cols, original_columns


def subtract_background(numeric_raw, numeric_plate):
    """Subtract the plate background from the raw data."""
    return numeric_raw.subtract(numeric_plate, fill_value=0)


def subtract_blank_row(df):
    """Subtract the first row (blank) from the entire DataFrame."""
    blank_row = df.iloc[[0]].values[0]
    return df.apply(lambda row: row - blank_row, axis=1)


def recombine_data(numeric_data, non_numeric_data, original_columns):
    """Recombine numeric, 'Well\nCol', and non-numeric columns in the original order."""
    combined_df = pd.concat([non_numeric_data, numeric_data], axis=1)

    # Reorder columns based on original column order
    # combined_df = combined_df.reindex(columns=original_columns)
    return combined_df


def group_and_calculate(df, operation='mean', group_size=4):
    """Group numeric data by a defined number of rows and calculate the specified operation. Should only be used on
    separated dataframe (i.e., numeric columns only)."""

    if operation == 'mean':
        grouped_df = df.groupby(df.index // group_size).mean()
    elif operation == 'std':
        grouped_df = df.groupby(df.index // group_size).std()
    else:
        grouped_df = df.groupby(df.index // group_size)

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
    numeric_cols_raw, non_numeric_cols_raw, original_columns_raw = separate_columns(raw_df)
    numeric_cols_plate, non_numeric_cols_plate, _ = separate_columns(plate_data)

    # Subtract plate background and blank row
    plate_background_removed = subtract_background(numeric_cols_raw, numeric_cols_plate)
    blank_removed = subtract_blank_row(plate_background_removed)

    # Recombine columns, respecting original order
    final_plate = recombine_data(blank_removed, non_numeric_cols_raw, original_columns_raw)

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

    # Create a new column for the row labels (A-H) and the column numbers (1-12)
    df['Row'] = df['Row/Col'].str[0]  # Extract row letter
    df['Col'] = df['Row/Col'].str[1:].astype(int)  # Extract column number

    # Pivot the DataFrame to a format suitable for a heatmap
    heatmap_data = df.pivot(index='Row', columns='Col', values=value_col)

    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='coolwarm', cbar=True, ax=ax)

    ax.set_title(title)


def plot_line(df, x_col_start, x_col_end, ax, title="Absorbance Spectra", samples_start=0, samples_end=4,
              wavelength_range=(220, 1000),
              ylim=(-1.0, 2)):
    """Plot absorbance spectra for the selected number of samples."""
    x = [int(i) for i in df.columns[x_col_start:x_col_end].values]  # Wavelength values
    y = [df.iloc[i, x_col_start:x_col_end].values for i in range(samples_start, samples_end)]  # Absorbance values

    for i in range(samples_start, samples_end):
        ax.plot(x, df.iloc[i, x_col_start:x_col_end].values, label=f'{df.iloc[i, 0]}')  # Use the index as the label

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


def main_3():
    path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\18-Sep-2024\Plate 2a.csv"
    data_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\18-Sep-2024\240919_1305.csv"
    out_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\18-Sep-2024"

    data = load_data_new(data_path)
    plate_2a = load_data_new(path)

    data_corrected = separate_subtract_and_recombine(data, plate_2a)

    # # Set up subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

    plot_line(data,
              x_col_start=1,
              x_col_end=data.shape[1],
              ax=axes[0, 0],
              title="Test",
              samples_start=0,
              samples_end=data.shape[0],
              wavelength_range=(220, 320),
              ylim=(-1, 3.5)
              )

    plot_line(data_corrected,
              x_col_start=1,
              x_col_end=data.shape[1],
              ax=axes[0, 1],
              title="Test",
              samples_start=0,
              samples_end=data.shape[0],
              wavelength_range=(220, 320),
              ylim=(-1, 3.5)
              )

    axes[0, 0].set_xticks(axes[0, 0].get_xticks()[::10])
    axes[0, 1].set_xticks(axes[0, 1].get_xticks()[::10])

    plot_heatmap(data, 280, "Heatmap of Plate 2a Background at 280 nm", ax=axes[1, 0])
    plot_heatmap(data_corrected, 280, "Heatmap of Plate 2a Background at 280 nm", ax=axes[1, 1])

    plt.savefig(out_path + "\plots.png")


def main_4():  # LSR curve fitting and linear regression model code
    # Specify path locations
    plate_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\18-Sep-2024\Plate 2a.csv"
    data_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\18-Sep-2024\240919_1305.csv"
    out_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\18-Sep-2024"

    # Load raw data and plate background
    data = load_data_new(data_path)
    plate_2a = load_data_new(plate_path)

    # Load separate spectra and conc data
    styrene_spectrum = load_data_new(
        r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Styrene & PS Cuvette Specs\PRD Plate Reader Specs\styrene 0.025 mgmL.csv")
    polystyrene_spectrum = load_data_new(
        r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Styrene & PS Cuvette Specs\PRD Plate Reader Specs\polystyrene 0.250 mgmL.csv")
    concs_df = load_data(
        r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\18-Sep-2024\Volumes 18-Sep Duplicated.csv")

    # Process data to remove plate and blank backgrounds
    data_corrected = separate_subtract_and_recombine(data, plate_2a)

    # Get numeric columns (spectrum curves) from component spectra
    num_styrene, non_numeric_cols, original_columns = separate_columns(styrene_spectrum)
    num_polystyrene, _, _ = separate_columns(polystyrene_spectrum)

    # Convert the Dataframes to NumPy arrays for easier calculations
    styrene_spectrum = num_styrene.values[0]  # 1D array for styrene absorbance
    polystyrene_spectrum = num_polystyrene.values[0]  # 1D array for polystyrene absorbance

    # Slice the data to restrict range to where most changes occur (40:101)
    range_start = 40
    range_end = 101
    styrene_spectrum = styrene_spectrum[range_start:]
    polystyrene_spectrum = polystyrene_spectrum[range_start:]

    # Initialize lists to store products of curve regression fitting
    x = []  # known component ratios
    y = []  # component ratios predicted from linear regression coefficients
    styrene_components_pred = []
    styrene_components_actual = []
    ps_components_pred = []
    ps_components_actual = []

    # For all sample spectra, performs regression fitting of component spectra and generates the coefficients for later use in training linear model
    for i in range(0, data_corrected.shape[0]):  #

        unknown = data_corrected.iloc[i,
                  1:].values  # 1D array for the unknown mixture absorbance at index i (represents each well read vertically)
        unknown_spectrum = unknown[range_start:]  # Slice to appropriate range

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

        styrene_components_pred.append(c_styrene_opt)
        styrene_components_actual.append(concs_df.iloc[i, 0] * .025 / 300)

        ps_components_pred.append(c_polystyrene_opt)
        ps_components_actual.append(concs_df.iloc[i, 1] * .25 / 300)

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

        # <editor-fold desc="For printing results of curve fitting.">
        print(f"Coefficient of styrene: {c_styrene_opt: .4f}")
        print(f"Coefficient of polystyrene: {c_polystyrene_opt: .4f}")
        print(f"Ratio of c(Styr) to c(p[Styr]): {ratio_pred: .4f}, {ratio_actual: .4f}")
        print(f"R Squared Value: {R_squared: .2f}")
        # </editor-fold>

        # <editor-fold desc="For plotting each fitted spectrum against the component spectra.">
        # fig, ax = plt.subplots(figsize=(8, 5))
        #
        # wavelengths = num_styrene.columns.astype(float)[40:101]
        #
        # plt.plot(wavelengths, unknown_spectrum, label=f'Mixture Spectrum for {concs_df.iloc[i, 0]} uL S, {concs_df.iloc[i, 1]} uL PS', color='black')
        # plt.plot(wavelengths, fitted_spectrum, label=f'Fitted Spectrum S/PS: {ratio_pred: .2f} pred vs {ratio_actual: .2f} actual', linestyle='--', color='blue')
        # plt.plot(wavelengths, styrene_spectrum, label='Styrene Spectrum', linestyle='-.', color='red')
        # plt.plot(wavelengths, polystyrene_spectrum, label='Polystyrene Spectrum', linestyle='-.',
        #          color='green')
        #
        # plt.xlabel('Wavelength (nm)')
        # plt.ylabel('Absorbance')
        # plt.legend(loc='best', fontsize=8)
        #
        # plt.savefig(rf"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\18-Sep-2024\Fitted Spectra\index {i}.png")
        # </editor-fold>

    # Convert x and y to NumPy arrays
    x = np.array(x)
    y = np.array(y)

    # <editor-fold desc="For plotting scatterplots of predicted vs actual fraction outputs">
    # Plot the results
    # fig, ax = plt.subplots(figsize=(8, 5))
    #
    # plt.scatter(styrene_components_actual, styrene_components_pred, color='black', label='Styrene', s=25)
    # plt.scatter(ps_components_actual, ps_components_pred, color='red', label='Polystyrene', s=25)
    #
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(True)
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['top'].set_visible(False)
    #
    # # Change spine width
    # for axis in ['top', 'bottom', 'left', 'right']:
    #     ax.spines[axis].set_linewidth(0.5)
    #
    # ax.minorticks_on()
    # ax.tick_params(axis='both', which='both', direction='in', pad=10)
    #
    # ax.set_title("Actual vs Predicted Spectral Fractions in Styrene/PS Mixtures")
    # ax.set_xlabel("Actual Fraction")
    # ax.set_ylabel("Predicted Spectral Fraction")
    #
    # ax.set_xlim(left=0)
    # ax.set_ylim(bottom=0)
    #
    # ax.grid(True, linestyle='-', linewidth=0.2, which='major', axis='both')
    # ax.legend(loc='best', fontsize=8)

    # plt.savefig(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\18-Sep-2024\PS Styr Conc Fraction plot test.png")
    # </editor-fold>

    # Split the data into training/testing sets
    x_train = np.array(styrene_components_pred[:-20]).reshape(-1, 1)
    x_test = np.array(styrene_components_pred[-20:]).reshape(-1, 1)

    # x_train = x_train[~np.isnan(x_train).any(axis=1)]
    # x_test = x_test[~np.isnan(x_test).any(axis=1)]

    # Split the targets into training/testing sets
    y_train = styrene_components_actual[:-20]
    y_test = styrene_components_actual[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    # The coefficients
    print("Equation: \n", f"y = {regr.coef_[0]: .4f}x + {regr.intercept_: .4f}")
    # The mean squared error
    print("Mean squared error: %.4f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.4f" % r2_score(y_test, y_pred))

    # <editor-fold desc="For plotting results of training and test y values versus the test set of x values">
    fig, ax = plt.subplots(figsize=(8, 5))

    # Test values vs predicted values (i.e., LOBF)
    plt.plot(x_test, y_pred, color='red', label=f"Expected Fit y = {regr.coef_[0]: .4f}x + {regr.intercept_: .4f}",
             linewidth="1", zorder=0)

    # Test values vs actual values
    plt.scatter(x_test, y_test, color='black', label='Test Data', s=25)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)

    # Change spine width
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', pad=10)

    ax.set_title("Predicted Spectral Fraction versus Styrene Concentration in Styrene/PS Mixtures")
    ax.set_xlabel("Predicted Spectral Fraction (%)")
    ax.set_ylabel("Styrene Concentration (mg/mL)")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    ax.grid(True, linestyle='-', linewidth=0.2, which='major', axis='both')
    ax.legend(loc='best', fontsize=8)

    plt.savefig(
        r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\skl linear model from 260nm onward.png")
    # </editor-fold>

    # Read in previous data for testing the model - uses old reader data so not best comparison
    old_data = load_data(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\12-Sep-2024\PS Sty Mixtures 12-Sep Blank and Background Corrected.csv")
    vols = load_data(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\Volumes_Duplicated.csv")

    # Extract sample curve and conc
    sample = old_data.iloc[6, 3:][range_start:]
    styrene_conc = vols.iloc[6, 0]*.025/300

    # Objective function
    def residuals(coeffs):
        c_styrene, c_polystyrene = coeffs
        combined_spectrum = c_styrene * styrene_spectrum + c_polystyrene * polystyrene_spectrum
        return np.sum((sample - combined_spectrum) ** 2)  # Sum of squared residuals

    # Minimize the residuals to find the best coefficients
    result = minimize(residuals, initial_guess, bounds=bounds)

    # Extract the fitted coefficients (concentrations)
    c_styrene_opt, c_polystyrene_opt = result.x

    c_styrene_opt = np.array([c_styrene_opt]).reshape(-1,1)

    # Make predictions using the testing set
    y_pred = regr.predict(c_styrene_opt)

    print(y_pred)


if __name__ == "__main__":
    main_4()
