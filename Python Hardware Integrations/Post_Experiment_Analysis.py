import matplotlib as mpl
from cycler import cycler
from scipy.optimize import curve_fit
import time as tm
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import AutoMinorLocator
from datetime import time, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tpot import TPOTRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from scipy.optimize import minimize_scalar
import os
import joblib

# Set plotting parameters globally
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['font.family'] = 'Times New Roman'

# Define a custom color palette
custom_colors = [
    '#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
    '#DDCC77', '#CC6677', '#882255', '#AA4499',  # Original palette
    '#661100', '#6699CC', '#AA4466', '#4477AA', '#228833',
    '#66CCEE', '#EEDD88', '#EE6677', '#AA3377', '#BBBBBB',
    '#333333', '#FFDD44', '#9988CC', '#66AA77', '#117755'
]

# Set the color cycle using plt.rc
plt.rc('axes', prop_cycle=cycler('color', custom_colors))

current_directory = os.getcwd()


def log_msg(message):
    """Log a message with a timestamp."""
    current_time = tm.strftime("%Y-%m-%d %H:%M:%S", tm.localtime())
    print(f"[{current_time}] {message}")


def load_data_new(path: str, start_wavelength: int = 220, end_wavelength: int = 1000) -> pd.DataFrame:
    """
    Loads a CSV file without headers and assumes the first column contains identifiers.
    Renames the columns to include 'Row/Col' and a range of wavelength values.

    :param path: The file path of the CSV to load.
    :param start_wavelength: The starting wavelength for column renaming.
    :param end_wavelength: The ending wavelength for column renaming.
    :return pd.DataFrame: A pandas DataFrame with updated column names.
    """
    try:
        df = pd.read_csv(path, header=None)
        df.columns = ['Row/Col'] + list(range(start_wavelength, end_wavelength + 1))
        return df
    except FileNotFoundError as e:
        log_msg(f"Error: File not found - {e}")
    except pd.errors.EmptyDataError as e:
        log_msg(f"Error: Empty file - {e}")
    except Exception as e:
        log_msg(f"Error loading file: {e}")
    return pd.DataFrame()


def load_data(path_input: str) -> pd.DataFrame:
    """
    Load CSV data into a Pandas DataFrame.
    Does not modify column headings or perform any data cleaning.

    :param path_input: The file path of the CSV to load.
    :return pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    try:
        return pd.read_csv(path_input)
    except FileNotFoundError as e:
        log_msg(f"Error: File not found - {e}")
    except pd.errors.EmptyDataError as e:
        log_msg(f"Error: Empty file - {e}")
    except Exception as e:
        log_msg(f"Error loading file: {e}")
    return pd.DataFrame()  # Return an empty DataFrame on failure


def separate_columns(df: pd.DataFrame) -> tuple:
    """
    Separate numeric and non-numeric columns from a DataFrame.

    :param df: The DataFrame to separate.
    :return tuple: Three items - numeric columns (DataFrame), non-numeric columns (DataFrame), and the original column order (list).
    """
    numeric_cols = df.select_dtypes(include='number')
    non_numeric_cols = df.select_dtypes(exclude='number')
    original_columns = df.columns.tolist()
    return numeric_cols, non_numeric_cols, original_columns


def recombine_data(numeric_data: pd.DataFrame, non_numeric_data: pd.DataFrame, original_columns: list) -> pd.DataFrame:
    """
    Recombine numeric and non-numeric columns back into their original order.

    :param numeric_data: The numeric data.
    :param non_numeric_data: The non-numeric data.
    :param original_columns: The original column order.

    :return pd.DataFrame: The recombined DataFrame with columns in the original order.
    """
    combined_df = pd.concat([non_numeric_data, numeric_data], axis=1)
    combined_df = combined_df.reindex(columns=original_columns)
    return combined_df


def separate_subtract_and_recombine(raw_df: pd.DataFrame, plate_data: pd.DataFrame,
                                    blank_index: int = 0) -> pd.DataFrame:
    """
    Separate numeric and non-numeric columns, subtract the plate background and blank row,
    and recombine all columns into their original order.

    :param raw_df: The raw data containing both numeric and non-numeric columns.
    :param plate_data: The plate background data.
    :param blank_index: The index of the row to use as the blank for correction.

    :return pd.DataFrame: The fully corrected DataFrame.
    """
    # Separate numeric and non-numeric columns
    numeric_cols_raw, non_numeric_cols_raw, original_columns_raw = separate_columns(raw_df)
    numeric_cols_plate, non_numeric_cols_plate, _ = separate_columns(plate_data)

    # Subtract plate background and blank row
    plate_corrected_data = numeric_cols_raw - numeric_cols_plate

    # Blank correction (optimized with vectorized subtraction)
    blank_row = plate_corrected_data.iloc[blank_index]
    blank_corrected_data = plate_corrected_data.subtract(blank_row, axis=1)

    # Recombine columns, maintaining original order
    final_plate = recombine_data(blank_corrected_data, non_numeric_cols_raw, original_columns_raw)

    return final_plate


def time_difference(time1, time2):
        """
        Calculate the time difference in minutes between two datetime.time objects,
        handling cases where the time spans overnight.
        """
        # Convert time1 and time2 to timedelta objects
        t1_delta = timedelta(hours=time1.hour, minutes=time1.minute, seconds=time1.second)
        t2_delta = timedelta(hours=time2.hour, minutes=time2.minute, seconds=time2.second)

        # Handle overnight case
        if t2_delta < t1_delta:
            t2_delta += timedelta(days=1)  # Add 24 hours to time2

        # Calculate the difference in minutes
        return (t2_delta - t1_delta).total_seconds() / 60


def plot_transmittance(plot_type, x_data, y_data, y_err=None, labels=None, title=None, xlabel=None, ylabel=None, save_name=None, out_path="."):
    """
    Generalized function to plot transmittance data.

    Args:
        plot_type (str): Type of plot ("individual" or "averaged").
        x_data (list or array): Data for the x-axis.
        y_data (list or DataFrame): Data for the y-axis. For "individual", it should be a DataFrame.
        y_err (list or array, optional): Error bars for the y-axis (only for "averaged"). Default is None.
        labels (list, optional): Labels for each plot line (only for "individual"). Default is None.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_name (str): File name to save the plot.
        out_path (str): Output path for saving the plot.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "individual":
            # Iterate over rows in y_data (assumed to be DataFrame) and plot each individually
            for i, (index, row) in enumerate(y_data.iterrows()):
                label = labels[i] if labels else None  # No label if labels=None
                ax.plot(x_data, row.values, 'o-', markersize=6, linewidth=1.5, label=label)

        elif plot_type == "averaged":
            # Plot with error bars
            ax.errorbar(x_data, y_data, yerr=y_err, ecolor='gray',
                        capsize=4, elinewidth=1, markeredgewidth=1, markersize=6, color="#41424C")
            ax.plot(x_data, y_data, 'o-', markersize=6, linewidth=1.5, color="#41424C", label='Averaged Data')

        # Customize axis labels and title
        ax.set_xlabel(xlabel, fontsize=14, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=14, labelpad=10)
        ax.set_title(title, fontsize=16, pad=15)

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        # Adjust tick parameters for readability
        ax.tick_params(axis='both', which='both', labelsize=12, width=1, length=5)

        # Add legend only if labels are provided
        if labels is not None:
            ax.legend(loc='best', fontsize=12, frameon=False)

        # Add grid
        ax.grid(False)

        # Apply tight layout and save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, save_name), dpi=300)  # High DPI for publication-quality
        plt.close()

    except Exception as e:
        print(f"Error while plotting: {e}")  # Replace with log_msg(e) if using a logging system


def first_below_threshold_index(row):
        # Iterate over the row with index and return the first index where value < threshold
        for idx, value in enumerate(row):
            if value < 50:
                return idx
        return None  # Return None if no values are below the threshold in the row


def fit_sigmoidal_and_plot(temperature, transmittance):
    def sigmoidal(x, A1, A2, x0, dx):
        return A2 + (A1 - A2) / (1 + np.exp((x - x0) / dx))

    global inflection_temps
    inflection_temps = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (index, row) in enumerate(transmittance.iterrows()):

        # Initial guess and bounds
        p0 = [np.max(row), np.min(row), np.mean(temperature), 1.0]
        bounds = ([0, 0, min(temperature), 0.1], [110, 110, max(temperature), 10])

        try:
            # Fit the curve
            popt, pcov = curve_fit(sigmoidal, temperature, row.values[:len(temperature)], p0=p0, bounds=bounds)
            A1, A2, x0, dx = popt

            # Generate fitted curve and find inflection point
            x_fine = np.linspace(min(temperature), max(temperature), 500)
            fitted_transmittance = sigmoidal(x_fine, *popt)
            dy_dx = np.gradient(fitted_transmittance, x_fine)
            inflection_index = np.argmin(dy_dx)
            inflection_temp = x_fine[inflection_index]
            inflection_temps.append(inflection_temp)

            # ax.plot(x_fine, dy_dx, linewidth=1.5, linestyle='--', color="#41424C", alpha=0.5)

            # Plot the fitted curve
            ax.plot(x_fine, fitted_transmittance, linewidth=1.5, linestyle='--', color="red", alpha=0.5)
            ax.axvline(inflection_temp, color="green", linestyle="--", linewidth=1, alpha=0.5,
                       label=f'Inflection Point: {inflection_temp: .1f} °C')

            ax.plot(temperature, row.values[:len(temperature)], 'o-', markersize=6, linewidth=1.5)

        except (RuntimeError, ValueError) as e:
            log_msg(f"Curve fitting failed for dataset {i}: {e}")
            continue

    # Customize axis labels and title
    ax.set_xlabel("Temperature (°C)", fontsize=14, labelpad=10)
    ax.set_ylabel("Average Transmittance (%)", fontsize=14, labelpad=10)
    ax.set_title("Turbidity Curves for Transmittance vs. Temperature for p(NIPAM), NaCl, and HCl in Water", fontsize=16, pad=15)

    # Adjust ticks and add minor locators
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='both', labelsize=12, width=1, length=5)

    # Add legend
    ax.legend(loc='best', fontsize=12, frameon=False)

    # Disable grid
    ax.grid(False)

    # Apply tight layout
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "sigmoidal_fit_transmittance.png"), dpi=300)  # Save high-DPI image


def prepare_data(concentrations, temperature_values, test_size=0.2, random_state=42):
    """Prepare and split data into training and testing sets."""
    X = np.array(concentrations)
    y = np.array(temperature_values)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_tpot_model(X_train, y_train, generations=10, population_size=75, scoring="r2", random_state=42):
    """
    Train and save a TPOTRegressor model.

    Args:
        X_train (array-like): Training data for features.
        y_train (array-like): Training data for the target variable.
        generations (int, optional): Number of generations to run the optimization. Default is 10.
        population_size (int, optional): Number of individuals in the population. Default is 75.
        scoring (str, optional): Scoring metric to optimize. Default is "r2".
        random_state (int, optional): Seed for reproducibility. Default is 42.
        out_path (str, optional): Path to save the trained model. Default is the current directory.

    Returns:
        tpot (TPOTRegressor): Trained TPOTRegressor model.
    """
    try:
        # Initialize and train the TPOT model
        tpot = TPOTRegressor(
            generations=generations,
            population_size=population_size,
            verbosity=2,
            random_state=random_state,
            scoring=scoring
        )
        tpot.fit(X_train, y_train)

        return tpot

    except Exception as e:
        print(f"An error occurred while training the model: {e}")
        return None


def evaluate_model(model, X_test, y_test, out_path):
    """Evaluate model performance and generate prediction plot."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Log metrics
    log_msg(f"Optimized model R²: {r2:.4f}")
    log_msg(f"Optimized model MSE: {mse:.4f}")

    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, y_test, marker="o", color="#41424C", s=60)
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Ideal Prediction Line")

    ax.set_xlabel("Predicted Temperature (°C)", fontsize=14)
    ax.set_ylabel("Actual Temperature (°C)", fontsize=14)
    ax.set_title("Actual vs. Predicted Temperature", fontsize=16)
    ax.text(0.05, 0.9, f'R² = {r2:.4f}\nMSE = {mse:.4f}', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "predicted_vs_actual.png"), dpi=300)
    plt.close()


def export_pipeline(model, out_path, filename="tpot_pipeline.py"):
    """Export TPOT pipeline to file."""
    pipeline_path = os.path.join(out_path, filename)
    model.export(pipeline_path)
    log_msg(f"Pipeline exported to {pipeline_path}")


def generate_model_insights(model, X_train, X_test, y_test, insight_out_path, feature_names):
    """Generate all model insight visualizations."""
    os.makedirs(insight_out_path, exist_ok=True)

    # Partial Dependence Plots
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        PartialDependenceDisplay.from_estimator(model.fitted_pipeline_, X_train, features=[0, 1, 2], ax=ax)
        ax.set_title("Partial Dependence Plots", fontsize=16)
        plt.savefig(os.path.join(insight_out_path, "partial_dependence.png"), dpi=300)
        plt.close()
    except Exception as e:
        log_msg(f"Partial Dependence Error: {e}")

    # Permutation Importance
    try:
        result = permutation_importance(model.fitted_pipeline_, X_test, y_test, n_repeats=30, random_state=42)
        sorted_idx = result.importances_mean.argsort()[::-1]

        plt.figure(figsize=(12, 8))
        plt.bar(range(len(sorted_idx)), result.importances_mean[sorted_idx], color="#41424C")
        plt.xticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.title("Feature Importance via Permutation", fontsize=16)
        plt.savefig(os.path.join(insight_out_path, "permutation_importance.png"), dpi=300)
        plt.close()
    except Exception as e:
        log_msg(f"Permutation Importance Error: {e}")

    # 3D Interaction Contour Plots (three plots)
    for fixed_var_index, fixed_var_name in enumerate(feature_names):
        try:
            # Get the fixed value as the mean of the training data for the fixed variable
            fixed_value = np.mean(X_train[:, fixed_var_index])

            # Determine the other two variables to vary
            varying_indices = [i for i in range(len(feature_names)) if i != fixed_var_index]
            var1_index, var2_index = varying_indices

            # Create ranges for the varying variables
            var1_range = np.linspace(X_train[:, var1_index].min(), X_train[:, var1_index].max(), 50)
            var2_range = np.linspace(X_train[:, var2_index].min(), X_train[:, var2_index].max(), 50)

            # Generate meshgrid for the varying variables
            var1_mesh, var2_mesh = np.meshgrid(var1_range, var2_range)

            # Create the grid points with the fixed variable set to its mean
            grid_points = np.zeros((var1_mesh.size, X_train.shape[1]))
            grid_points[:, var1_index] = var1_mesh.ravel()
            grid_points[:, var2_index] = var2_mesh.ravel()
            grid_points[:, fixed_var_index] = fixed_value

            # Predict using the model
            predictions = model.fitted_pipeline_.predict(grid_points).reshape(var1_mesh.shape)

            # Create the contour plot
            fig, ax = plt.subplots(figsize=(12, 8))
            contour = ax.contourf(var1_mesh, var2_mesh, predictions, cmap='viridis', levels=20)
            cbar = plt.colorbar(contour)
            cbar.set_label("Predicted LCST (°C)", fontsize=12)

            # Set axis labels and title
            ax.set_xlabel(feature_names[var1_index], fontsize=14, labelpad=10)
            ax.set_ylabel(feature_names[var2_index], fontsize=14, labelpad=10)
            ax.set_title(
                f"LCST as a Function of {feature_names[var1_index]} and {feature_names[var2_index]}\n({fixed_var_name} Fixed at {fixed_value:.3f})",
                fontsize=16,
                pad=15,
            )

            plt.tight_layout()
            plt.savefig(
                os.path.join(insight_out_path, f"interaction_contour_plot_{fixed_var_name}_fixed.png"),
                dpi=300,
            )
            plt.close()
        except Exception as e:
            log_msg(f"Contour Plot Error for fixed {fixed_var_name}: {e}")


def optimize_concentration(model, target_lcst, bounds=(0, 10), fixed_values=[0.142/100, 1.25/100000]):
    """Find optimal polymer concentration for target LCST."""

    def predict_lcst(concentration):
        return model.predict(np.array([[concentration, *fixed_values]]))[0]

    result = minimize_scalar(
        lambda x: abs(predict_lcst(x) - target_lcst),
        bounds=bounds,
        method='bounded'
    )

    if result.success:
        log_msg(f"Optimal concentration: {result.x:.4f} mg/mL")
        return result.x
    else:
        log_msg("Optimization failed")
        return None


def generate_volumes_csv(optimal_conc, out_path, standard_conc=1000, total_vol=1000):
    """Generate CSV with component volumes for optimal concentration."""
    component_vol = (optimal_conc / standard_conc) * total_vol
    solvent_vol = total_vol - component_vol

    df = pd.DataFrame({
        'Component 1 (µL)': [component_vol],
        'Solvent (µL)': [solvent_vol]
    })

    timestamp = tm.strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(out_path, f"solution_volumes_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    log_msg(f"Volumes saved to {csv_path}")
    return df


if __name__ == "__main__":
    with open(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\LCST\23-Jan full plate + salt + HCl\32.5 C Predicted Mixture\temperature_data_2025-01-30 15_59_41.json") as f:
        f = json.load(f)
        averages = f["averages"]
        std_devs = f["std_devs"]
        temps1_plotting = f["temps1_plotted"]
        time_stamps = f["measurement_timestamps"]
        temps1 = f["temps1"]
        temps2 = f["temps2"]

    out_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\LCST\23-Jan full plate + salt + HCl\32.5 C Predicted Mixture"
    data_paths = []
    folder_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\LCST\23-Jan full plate + salt + HCl\32.5 C Predicted Mixture\abs_spectra"
    volumes_csv = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\LCST\23-Jan full plate + salt + HCl\32.5 C Predicted Mixture\Duplicated_Volumes.csv"
    volumes_df = load_data(volumes_csv)
    measurement_times = []

    # Iterate over files in target folder
    for idx, filename in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        # Check if it’s a file (and not a directory)
        if os.path.isfile(file_path):
            if idx == 0:
                # Save the first file as plate background path
                plate_background_path = file_path
            else:
                # Save remaining files to data_paths
                data_paths.append(file_path)

                # Extract the file name
                file_name = os.path.basename(file_path)

                # Extract the time part (last 4 characters before ".csv")
                time_str = file_name.split('_')[-1].split('.')[0]

                # Convert to a datetime.time object
                measurement_time = datetime.strptime(time_str, "%H%M").time()
                measurement_times.append(measurement_time)

    # Post-process data to calculate transmittance and plot results
    abs_std_dev = []
    trans_dfs = []
    transmittance_dir = os.path.join(out_path, 'transmittance spectra')
    os.makedirs(transmittance_dir, exist_ok=True)
    full_absorbance_list = []

    for path in data_paths:
        try:
            plate = load_data_new(plate_background_path)
            data = load_data_new(path)
            corrected_array = separate_subtract_and_recombine(data, plate, 0)[600].to_numpy()
            corrected_full = separate_subtract_and_recombine(data, plate, 0).to_numpy()
            full_absorbance_list.append(corrected_full)

            transmittance_array = 10 ** (-corrected_array) * 100
            transmittance_df = pd.DataFrame(transmittance_array)
            trans_dfs.append(transmittance_df)
            transmittance_filename = os.path.join(transmittance_dir, f"transmittance_{os.path.basename(path)}")
            transmittance_df.to_csv(transmittance_filename, index=False)

            average = np.average(transmittance_array)
            std = np.std(transmittance_array)
            abs_std_dev.append((average, std))
        except Exception as e:
            log_msg(f"Error processing path {path}: {e}")

    # Concatenate all transmittance data into a single DataFrame
    stacked_transmittance_df = pd.concat(trans_dfs, axis=1)

    # Save the combined transmittance DataFrame
    stacked_filename = os.path.join(transmittance_dir, "stacked_transmittance.csv")
    stacked_transmittance_df.to_csv(stacked_filename, index=False)

    # Extract averages and standard deviations for plotting
    averages = [item[0] for item in abs_std_dev]
    std_devs = [item[1] for item in abs_std_dev]

    try:
        # Apply the function to each row in stacked_transmittance_df
        first_below_threshold = stacked_transmittance_df.apply(first_below_threshold_index, axis=1)

        log_msg(stacked_transmittance_df.shape[1])
        log_msg(len(temps1_plotting))

        temperature_values = []
        concentrations = []

        # Display the result
        for idx, value in enumerate(first_below_threshold):
            if pd.notna(value):  # Check if a valid integer index was found (not NaN)
                transmittance_value = stacked_transmittance_df.iloc[idx, int(value)]
                temperature = temps1_plotting[int(value)]

                # Append values for plotting
                temperature_values.append(temperature)
                concentrations.append([volumes_df.iloc[idx, 0] * (10 / 300), volumes_df.iloc[idx, 1] * ((1/100) / 300), volumes_df.iloc[idx, 2] * (1/10000 / 300)])

                log_msg(f"Row {idx}: Transmittance = {transmittance_value}, Temperature = {temperature}°C")
            else:
                log_msg(f"Row {idx}: No transmittance values below threshold")

    except Exception as e:
        log_msg(e)

    concentrations = np.array(concentrations)

    # Calls for each plot type
    # Plot individual transmittances over temperature
    plot_transmittance(
        plot_type="individual",
        x_data=temps1_plotting,
        y_data=stacked_transmittance_df.iloc[:, 1:],
        labels=[f"{round(volumes_df.iloc[i, 0] * (10 / 300), 2)} mg/mL" for i in range(len(stacked_transmittance_df))],
        title='Percent Transmittance at 600 nm of p(NIPAM), NaCl, and HCl in Water vs. Temperature',
        xlabel="Temperature (°C)",
        ylabel='Average Transmittance (%)',
        save_name="trans_versus_temp_individual.png",
        out_path=out_path
    )

    # Plot individual transmittances over time
    try:
        times = [time_difference(measurement_times[0], t) for t in measurement_times]
    except Exception as e:
        log_msg(e)

    plot_transmittance(
        plot_type="individual",
        x_data=times,
        y_data=stacked_transmittance_df,
        labels=[f"{round(volumes_df.iloc[i, 0] * (10 / 300), 2)} mg/mL" for i in range(len(stacked_transmittance_df))],
        title='Percent Transmittance at 600 nm of p(NIPAM), NaCl, and HCl in Water vs. Time',
        xlabel="Time (Seconds)",
        ylabel='Average Transmittance (%)',
        save_name="trans_versus_time_individual.png",
        out_path=out_path
    )

    # Plot averaged transmittances over temperature
    plot_transmittance(
        plot_type="averaged",
        x_data=temps1_plotting[:len(averages)//2],
        y_data=averages[:len(averages)//2],
        y_err=std_devs[:len(averages)//2],
        title='Percent Transmittance at 600 nm of p(NIPAM), NaCl, and HCl in Water vs. Temperature',
        xlabel="Temperature (°C)",
        ylabel='Average Transmittance (%)',
        save_name="trans_versus_temp_averaged.png",
        out_path=out_path
    )

    temperature = temps1_plotting[:len(temps1_plotting) // 2]
    transmittance = stacked_transmittance_df
    fit_sigmoidal_and_plot(temperature, transmittance)

    temperature_values_filtered = np.array([temp for temp in inflection_temps[4:] if temp is not None])

    print(concentrations.shape, temperature_values_filtered.shape)

    FEATURE_NAMES = ["Polymer", "NaCl", "HCl"]

    # Load and prepare data
    X_train, X_test, y_train, y_test = prepare_data(concentrations, temperature_values_filtered)

    # Train model
    model = train_tpot_model(X_train, y_train)

    # Evaluate and export
    evaluate_model(model, X_test, y_test, out_path)
    export_pipeline(model, out_path)

    # Generate insights
    generate_model_insights(model, X_train, X_test, y_test,
                            os.path.join(out_path, "insights"), FEATURE_NAMES)

    # Optimization and solution generation
    target = 32.5  # Target temperature
    optimal = optimize_concentration(model, target)
    if optimal:
        generate_volumes_csv(optimal, out_path)
