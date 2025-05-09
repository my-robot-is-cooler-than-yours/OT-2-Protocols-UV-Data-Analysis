import matplotlib as mpl
from cycler import cycler
from scipy.optimize import curve_fit
import time as tm
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from datetime import datetime
from matplotlib.ticker import AutoMinorLocator
from datetime import time, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tpot import TPOTRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from scipy.optimize import minimize_scalar
import os
from xgboost import XGBRegressor
import logging
from sklearn.svm import SVR
from scipy.stats import randint, uniform, reciprocal
from sklearn.model_selection import RandomizedSearchCV
from pandas.plotting import parallel_coordinates
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Define colours for the cmap (from cold to warm)
colormaps = {
    "subtle_cool_warm": [
        (0, 0, 0.5), (0.2, 0.4, 0.7), (0.7, 0.7, 0.7), (0.6, 0.4, 0.4), (0.5, 0.2, 0.2), (0.3, 0, 0)
    ],
    "scientific_cmap": [
        (0, 0, 0.5), (0.2, 0.4, 0.8), (0.75, 0.75, 0.75), (0.6, 0.4, 0.3), (0.5, 0.2, 0.2), (0.3, 0, 0)
    ],
    "pastel_cmap": [
        (0.2, 0.4, 0.7), (0.4, 0.6, 0.9), (0.8, 0.8, 0.8), (0.7, 0.5, 0.5), (0.6, 0.3, 0.3), (0.4, 0.1, 0.2)
    ],
    "monash_lab": [
        (0, 0.1, 0.6), (0.2, 0.5, 0.8), (0.7, 0.7, 0.7), (0.6, 0.4, 0.3), (0.5, 0.2, 0.1), (0.3, 0, 0)
    ],
    "deep_cmap": [
        (0, 0, 0.6), (0.2, 0.4, 0.9), (0.8, 0.8, 0.8), (0.7, 0.5, 0.4), (0.5, 0.2, 0.2), (0.3, 0, 0)
    ],
    "sunset_cmap": [
        (0, 0.2, 0.5), (0.3, 0.5, 0.7), (0.85, 0.8, 0.75), (0.7, 0.5, 0.4), (0.6, 0.3, 0.3), (0.4, 0, 0)
    ],
    "nasa_cmap": [
        (0, 0, 0.4), (0.1, 0.3, 0.8), (0.75, 0.75, 0.75), (0.7, 0.5, 0.4), (0.6, 0.3, 0.3), (0.3, 0, 0)
    ],
    "minimalist_cmap": [
        (0.3, 0.4, 0.6), (0.5, 0.6, 0.8), (0.8, 0.75, 0.75), (0.7, 0.5, 0.5), (0.6, 0.3, 0.3), (0.4, 0.1, 0.2)
    ],
    "ocean_lava": [
        (0, 0.2, 0.7), (0, 0.5, 0.9), (0.8, 0.8, 0.75), (0.7, 0.5, 0.4), (0.6, 0.3, 0.2), (0.3, 0, 0)
    ],
    "ice_fire": [
        (0.1, 0.2, 0.7), (0.3, 0.6, 0.9), (0.75, 0.75, 0.75), (0.7, 0.5, 0.4), (0.6, 0.3, 0.2), (0.3, 0.1, 0.2)
    ]
}

vibrant_colormaps = {
    "vibrant_cool_warm": [
        (0, 0, 1),  # Pure blue
        (0.2, 0.6, 1),  # Sky blue
        (0.85, 0.85, 0.85),  # Soft neutral gray
        (1, 0.6, 0.6),  # Light salmon
        (1, 0.2, 0.2),  # Bright red
        (0.6, 0, 0)  # Deep red
    ],
    "academic_cmap": [
        (0, 0, 0.9),  # Deep blue
        (0.2, 0.5, 1),  # Vivid blue
        (0.85, 0.85, 0.85),  # Light gray
        (1, 0.6, 0.6),  # Light pinkish-red
        (1, 0.3, 0.3),  # Scarlet red
        (0.7, 0, 0)  # Dark red
    ],
    "scientific_vibrant": [
        (0, 0.1, 1),  # Electric blue
        (0.3, 0.7, 1),  # Lighter sky blue
        (0.9, 0.9, 0.9),  # Soft white-gray
        (1, 0.5, 0.5),  # Warm pink
        (1, 0.1, 0.1),  # Bright crimson
        (0.5, 0, 0)  # Deep wine red
    ],
    "bold_cmap": [
        (0, 0.1, 0.9),  # Strong blue
        (0.3, 0.6, 1),  # Bright blue
        (0.85, 0.85, 0.85),  # Light neutral
        (1, 0.5, 0.4),  # Coral pink
        (1, 0.2, 0.1),  # Vivid red-orange
        (0.5, 0, 0)  # Blood red
    ],
    "modern_heatmap": [
        (0, 0.2, 1),  # Intense blue
        (0.4, 0.7, 1),  # Soft blue
        (0.9, 0.9, 0.9),  # Off-white
        (1, 0.6, 0.4),  # Light peach
        (1, 0.3, 0.2),  # Burnt red
        (0.7, 0, 0)  # Dark ruby red
    ],
    "pastel_vibrant": [
        (0.2, 0.5, 1),  # Sky blue
        (0.5, 0.8, 1),  # Light blue
        (0.9, 0.9, 0.9),  # Off-white
        (1, 0.6, 0.6),  # Soft pink
        (1, 0.4, 0.4),  # Blush red
        (0.8, 0, 0)  # Deep crimson
    ],
    "ocean_fire": [
        (0, 0.1, 1),  # Strong blue
        (0.3, 0.7, 1),  # Soft blue
        (0.85, 0.85, 0.85),  # Neutral white-gray
        (1, 0.6, 0.4),  # Light orange-pink
        (1, 0.3, 0.2),  # Vivid red
        (0.6, 0, 0)  # Dark red
    ],
    "monash_vibrant": [
        (0, 0.2, 0.9),  # Intense deep blue
        (0.3, 0.6, 1),  # Soft blue
        (0.85, 0.85, 0.85),  # Neutral
        (1, 0.5, 0.4),  # Warm coral
        (1, 0.2, 0.2),  # Bright red
        (0.6, 0, 0)  # Deep burgundy
    ],
    "lava_glacier": [
        (0, 0, 1),  # Pure blue
        (0.3, 0.6, 1),  # Bright sky blue
        (0.9, 0.9, 0.9),  # Near-white
        (1, 0.5, 0.4),  # Soft orange-pink
        (1, 0.2, 0.2),  # Intense red
        (0.7, 0, 0)  # Dark cherry red
    ],
    "fire_ice": [
        (0, 0.1, 1),  # Rich blue
        (0.3, 0.7, 1),  # Soft light blue
        (0.9, 0.9, 0.9),  # Almost white
        (1, 0.5, 0.4),  # Peach coral
        (1, 0.3, 0.2),  # Bright scarlet
        (0.6, 0, 0)  # Dark red
    ]
}


def create_cmap(name, dict):
    return LinearSegmentedColormap.from_list(name, dict[name], N=256)


# Set plotting parameters globally
mpl.rcParams['figure.dpi'] = 600
# mpl.rcParams['font.family'] = 'Arial'

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


def apply_custom_plot_style():
    """Applies specific stylistic choices to matplotlib plots."""
    plt.style.use('seaborn-v0_8-whitegrid')  # Start with a clean base

    # --- Global Font Settings ---
    # Use a common sans-serif font. Arial might not be available on all systems.
    # DejaVu Sans is a good default.
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']  # Or ['Arial'] if available
    plt.rcParams['font.weight'] = 'normal'  # Default weight
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title (if used)

    # --- Axes Settings ---
    plt.rcParams['axes.edgecolor'] = 'black'  # Color of the box outline
    plt.rcParams['axes.linewidth'] = 1.5  # Thicker axes lines/spines

    # --- Tick Settings ---
    plt.rcParams['xtick.direction'] = 'out'  # Ticks point inwards
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.size'] = 5  # Size of major ticks
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['xtick.major.width'] = 1.5  # Thickness of major ticks
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.labelsize'] = 11  # Font size for x-tick labels
    plt.rcParams['ytick.labelsize'] = 11  # Font size for y-tick labels
    # Set tick labels to appear bold (matching the image)
    # Note: This isn't a direct rcParam, often handled during plotting or requires custom Formatter
    # We will handle bold tick labels manually if needed, but often font choice suffices.

    # --- Grid Settings ---
    plt.rcParams['axes.grid'] = False  # Turn off the grid

    # --- Legend Settings ---
    plt.rcParams['legend.frameon'] = False  # No frame around the legend
    plt.rcParams['legend.fontsize'] = 10  # Font size for legend text

    # --- Scatter Plot Specifics (can be set during plotting) ---
    # plt.rcParams['scatter.marker'] = '.' # Default marker if needed globally

    # --- LaTeX Settings ---
    # Ensure LaTeX works correctly if installed
    # plt.rcParams['text.usetex'] = True # Uncomment if LaTeX is installed and desired


def log_msg(*messages, sep=' ', end='\n'):
    """Log messages with a timestamp. Behaves similarly to print() with additional timestamp."""
    current_time = tm.strftime("%Y-%m-%d %H:%M:%S", tm.localtime())
    message_str = sep.join(str(msg) for msg in messages)
    print(f"[{current_time}] {message_str}", end=end)


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


def load_config_data(json_path, volumes_csv_path):
    """Load JSON configuration data and volumes CSV"""
    with open(json_path) as f:
        json_data = json.load(f)

    volumes_df = load_data(volumes_csv_path)
    return json_data, volumes_df


def process_spectra_files(folder_path):
    """Process spectra folder to identify data files and measurement times"""
    plate_background_path = None
    data_paths = []
    measurement_times = []

    for idx, filename in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            if idx == 0:
                plate_background_path = file_path
            else:
                data_paths.append(file_path)
                time_str = filename.split('_')[-1].split('.')[0]
                measurement_time = datetime.strptime(time_str, "%H%M").time()
                measurement_times.append(measurement_time)

    return plate_background_path, data_paths, measurement_times


def calculate_transmittance(data_paths, plate_background_path, idx_start, idx_end, wavelength_start, wavelength_end,
                            out_path):
    """Calculate and save transmittance data for all paths"""
    trans_dfs = []
    abs_std_dev = []
    full_absorbance_list = []
    transmittance_dir = os.path.join(out_path, 'transmittance spectra')
    os.makedirs(transmittance_dir, exist_ok=True)

    for path in data_paths:
        try:
            plate = load_data_new(plate_background_path, wavelength_start, wavelength_end)
            data = load_data_new(path, wavelength_start, wavelength_end)
            corrected_array = separate_subtract_and_recombine(data, plate, 0)[600][idx_start:idx_end].to_numpy()
            corrected_full = separate_subtract_and_recombine(data, plate, 0).to_numpy()
            full_absorbance_list.append(corrected_full)

            transmittance_array = 10 ** (-corrected_array) * 100
            transmittance_df = pd.DataFrame(transmittance_array)
            trans_dfs.append(transmittance_df)

            # Save individual transmittance data
            transmittance_filename = os.path.join(
                transmittance_dir,
                f"transmittance_{os.path.basename(path)}"
            )
            transmittance_df.to_csv(transmittance_filename, index=False)

            # Calculate statistics
            average = np.average(transmittance_array)
            std = np.std(transmittance_array)
            abs_std_dev.append((average, std))

            # Convert full_absorbance_list to a DataFrame
            full_absorbance_df = pd.DataFrame(np.vstack(full_absorbance_list))
            full_absorbance_df = full_absorbance_df.iloc[:, 1:]
            full_absorbance_df.columns = range(wavelength_start, wavelength_end + 1)

        except Exception as e:
            log_msg(f"Error processing path {path}: {e}")

    return trans_dfs, abs_std_dev, full_absorbance_df


def find_inflection_temps(temperature, transmittance):
    def sigmoidal(x, A1, A2, x0, dx):
        return A2 + (A1 - A2) / (1 + np.exp((x - x0) / dx))

    inflection_temps = []
    dx_values = []

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
            dx_values.append(dx)

        except Exception as e:
            log_msg(e)

    return inflection_temps, dx_values


def find_onset_temps(temperature, transmittance,
                     smooth_sigma=2.0, baseline_frac=0.1):
    """
    For each transmittance curve (row), find the temperature at the onset
    of cloudiness by the tangent‐intersection method.

    Returns
    -------
    onset_temps : list of float
        Estimated onset temperature for each curve.
    slopes : list of float
        Slope of the tangent at the inflection point.
    """
    onset_temps = []
    slopes = []

    T = np.array(temperature)
    n_points = len(T)

    for _, row in transmittance.iterrows():
        y = row.values[:n_points].astype(float)

        # 1. Smooth the curve to suppress noise
        y_smooth = gaussian_filter1d(y, sigma=smooth_sigma)

        # 2. Compute derivative
        dy_dT = np.gradient(y_smooth, T)

        # 3. Locate inflection: max absolute slope
        idx_inflect = np.argmax(np.abs(dy_dT))
        T_inflect = T[idx_inflect]
        slope = dy_dT[idx_inflect]

        # 4. Estimate baseline as average of the first baseline_frac of data
        n_base = max(int(n_points * baseline_frac), 2)
        baseline = np.mean(y_smooth[:n_base])

        # 5. Solve for onset: intersection of tangent and baseline
        y_inflect = y_smooth[idx_inflect]
        if slope == 0:
            T_onset = np.nan  # avoid divide-by-zero
        else:
            T_onset = T_inflect + (baseline - y_inflect) / slope

        onset_temps.append(T_onset)
        slopes.append(slope)

    return onset_temps, slopes


def prepare_all_data(json_path, volumes_csv_path, spectra_folder_path, out_path, idx_start, idx_end, wavelength_start,
                     wavelength_end):
    """Main data preparation function combining all steps"""
    # Load configuration data
    json_data, volumes_df = load_config_data(json_path, volumes_csv_path)

    concentrations = []
    for idx in range(volumes_df.shape[0]):
        concentrations.append([volumes_df.iloc[idx, 0] * ((10) / 300),
                               volumes_df.iloc[idx, 1] * ((1 / 100) / 300),
                               volumes_df.iloc[idx, 2] * ((1 / 10000) / 300)])
    concentrations = np.array(concentrations)

    # Process spectra files
    plate_background_path, data_paths, measurement_times = process_spectra_files(spectra_folder_path)

    # Calculate transmittance data
    trans_dfs, abs_std_dev, full_absorbance_list = calculate_transmittance(
        data_paths, plate_background_path, idx_start, idx_end, wavelength_start, wavelength_end, out_path
    )

    # Combine transmittance data
    transmittance_dir = os.path.join(out_path, 'transmittance spectra')
    stacked_transmittance_df = pd.concat(trans_dfs, axis=1)
    stacked_filename = os.path.join(transmittance_dir, "stacked_transmittance.csv")
    stacked_transmittance_df.to_csv(stacked_filename, index=False)

    # Extract statistics
    averages = [item[0] for item in abs_std_dev]
    std_devs = [item[1] for item in abs_std_dev]

    # Extract relevant JSON data
    temps1_plotting = json_data["temps1_plotted"]
    time_stamps = json_data["measurement_timestamps"]
    temps1 = json_data["temps1"]
    temps2 = json_data["temps2"]

    inflection_temps, dxs = find_inflection_temps(temps1_plotting[:len(temps1_plotting) // 2], stacked_transmittance_df)
    onset_temps, slopes = find_onset_temps(temps1_plotting[:len(temps1_plotting) // 2], stacked_transmittance_df)

    return {
        'stacked_transmittance_df': stacked_transmittance_df,
        'concentrations': concentrations,
        'averages': averages,
        'std_devs': std_devs,
        'measurement_times': measurement_times,
        'temps1_plotting': temps1_plotting,
        'volumes_df': volumes_df,
        'full_absorbance_list': full_absorbance_list,
        'temps1': temps1,
        'temps2': temps2,
        'time_stamps': time_stamps,
        'inflection_temps': inflection_temps,
        'onset_temps': onset_temps,
        'dx_values': dxs,
        'slopes': slopes,
    }


def plot_transmittance(plot_type, x_data, y_data, y_err=None, labels=None, title=None, xlabel=None, ylabel=None,
                       save_name=None, out_path="."):
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
        log_msg(f"Error while plotting {title}: {e}")  # Replace with log_msg(e) if using a logging system


def create_heatmap(
        data,
        num_initial_nans=4,
        reshape_shape=(8, 12),
        rows=None,
        columns=None,
        title="Measured LCST Values Per Well",
        filename="heatmap_bubble.png",
        figsize=(10, 6),
        cmap="coolwarm",
        edgecolors="#D3D3D3",
        s=1100,
        dpi=300,
        fontsize_labels=12,
        fontsize_title=16,
        colorbar_fontsize=10,
        **kwargs
):
    """
    Generates a bubble heatmap from 1D data array with configurable parameters.

    Parameters:
        data (array-like): Input 1D data array
        num_initial_nans (int): Number of initial values to set as NaN
        reshape_shape (tuple): Target shape for heatmap grid (rows, cols)
        rows/columns (list): Labels for rows/columns
        title/filename (str): Plot title and output path
        figsize (tuple): Figure dimensions
        cmap/edgecolors: Colormap and edge colors for bubbles
        s (int): Bubble size
        dpi (int): Output image resolution
        fontsize_*: Font size controls
        **kwargs: Additional arguments for matplotlib.scatter
    """
    inflection_temps = data.copy()
    inflection_temps[:num_initial_nans] = np.nan
    data_reshaped = inflection_temps.reshape(reshape_shape)

    # Generate labels if not provided
    rows = rows or [chr(ord('A') + i) for i in range(reshape_shape[0])]
    columns = columns or np.arange(1, reshape_shape[1] + 1)

    df = pd.DataFrame(data_reshaped, index=rows, columns=columns)

    fig, ax = plt.subplots(figsize=figsize)
    row_indices = np.arange(len(rows))
    col_indices = np.arange(len(columns))

    # Create plotting coordinates
    x, y = np.meshgrid(col_indices, row_indices)
    x, y = x.flatten(), y.flatten()
    values = df.to_numpy().flatten()

    # Normalize colormap excluding initial NaNs
    valid_values = values[num_initial_nans:]
    norm = plt.Normalize(
        vmin=np.nanmin(valid_values),
        vmax=np.nanmax(valid_values)
    )

    # Generate heatmap bubbles
    scatter = ax.scatter(
        x, y, s=s, c=values, cmap=cmap, norm=norm,
        edgecolors=edgecolors, **kwargs
    )

    # Configure axis appearance
    ax.set_xticks(col_indices)
    ax.set_xticklabels(columns, fontsize=fontsize_labels)
    ax.set_yticks(row_indices)
    ax.set_yticklabels(rows, fontsize=fontsize_labels)
    ax.tick_params(left=False, bottom=False)
    ax.set_title(title, fontsize=fontsize_title)
    ax.invert_yaxis()
    ax.set_frame_on(False)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.04)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()


def create_boxplot(
        data,
        num_initial_excluded=4,
        names=None,
        title="LCST Distribution Across All Samples",
        filename="box_plot.png",
        figsize=(6, 8),
        ylabel="LCST (°C)",
        yticks_count=10,
        ytick_format='%.2f',
        box_facecolor="#D3D3D3",
        median_color="grey",
        dpi=300,
        fontsize_labels=12,
        fontsize_title=16,
        fontsize_ylabel=14,
        label_rotation=45,
        **kwargs
):
    """
    Generates box plots from dataset(s) with configurable parameters.

    Parameters:
        data (array-like or list): Input data (1D array or list of arrays)
        num_initial_excluded (int): Number of initial values to exclude per dataset
        names (list): Optional names for x-axis labels
        title/filename (str): Plot title and output path
        figsize (tuple): Figure dimensions
        ylabel (str): Y-axis label text
        yticks_count (int): Number of y-axis ticks
        ytick_format (str): Format string for y-axis
        box_facecolor/median_color: Styling parameters
        dpi (int): Output image resolution
        fontsize_*: Font size controls
        label_rotation (int): Rotation angle for x-axis labels (degrees, default 45)
        **kwargs: Additional arguments for matplotlib.boxplot
    """
    # Convert input to list of datasets
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            datasets = [data]
        else:
            datasets = [data[i] for i in range(data.shape[0])]
    elif not isinstance(data, (list, tuple)):
        datasets = [data]
    else:
        datasets = data

    # Process datasets
    processed_data = []
    for dataset in datasets:
        inflection_temps = np.array(dataset).copy()[num_initial_excluded:]
        processed_data.append(inflection_temps.flatten())

    # Handle empty case
    if not processed_data:
        raise ValueError("No valid data provided")

    # Generate x-axis labels
    if names is None:
        names = ["All Wells"] if len(processed_data) == 1 else \
            [str(i + 1) for i in range(len(processed_data))]
    elif len(names) != len(processed_data):
        raise ValueError("Names length must match number of datasets")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Box plot styling
    boxprops = dict(facecolor=box_facecolor, color="grey")
    medianprops = dict(color=median_color, linewidth=2)

    # Create box plots
    ax.boxplot(
        processed_data,
        patch_artist=True,
        notch=False,
        boxprops=boxprops,
        medianprops=medianprops,
        **kwargs
    )

    # Axis configuration
    ax.set_xticks(np.arange(1, len(processed_data) + 1))
    ax.set_xticklabels(names, fontsize=fontsize_labels, rotation=label_rotation,
                       ha='right')  # Rotate labels and align right
    ax.set_ylabel(ylabel, fontsize=fontsize_ylabel)

    # Y-axis formatting
    all_values = np.concatenate(processed_data)
    y_min, y_max = np.nanmin(all_values), np.nanmax(all_values)
    ax.set_yticks(np.linspace(y_min, y_max, yticks_count))
    ax.yaxis.set_major_formatter(FormatStrFormatter(ytick_format))

    ax.set_title(title, fontsize=fontsize_title)
    plt.grid(False)

    # Adjust layout to accommodate rotated labels
    plt.tight_layout()

    plt.savefig(filename, dpi=dpi, bbox_inches='tight')  # Use bbox_inches='tight' for better spacing
    plt.close()


def plot_parallel_coords(x, y, z, lcst_values, title, xlabel, ylabel, zlabel, cmap=cm.seismic, save_name=None):
    """
    Plots a parallel-coordinates style figure with 4 axes:
      - Axis 0: x (e.g., [Polymer])
      - Axis 1: y (e.g., [NaCl])
      - Axis 2: z (e.g., [HCl])
      - Axis 3: LCST (dependent variable)

    Each axis is scaled independently so that its original range is preserved,
    and smooth Bezier curves are drawn connecting the axes for each sample.

    Args:
        x, y, z (array-like): Independent variable arrays.
        lcst_values (array-like): Dependent variable (LCST values).
        title (str): Title of the plot.
        xlabel (str): Label for the first axis.
        ylabel (str): Label for the second axis.
        zlabel (str): Label for the third axis.
        save_name (str, optional): Filename to save the figure.
    """
    # Combine data into an array of shape (N, 4)
    # Columns: 0 -> x, 1 -> y, 2 -> z, 3 -> LCST
    ys = np.column_stack([x, y, z, lcst_values])
    num_samples, num_dims = ys.shape  # Expecting num_dims == 4

    # Axis labels for x-axis ticks
    axis_labels = [xlabel, ylabel, zlabel, "LCST"]

    # -------------------------------
    # 1. Compute min, max, and range for each axis
    # -------------------------------
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    # Add 5% padding to each axis
    ymins -= dys * 0.05
    ymaxs += dys * 0.05
    dys = ymaxs - ymins

    # -------------------------------
    # 2. Transform axes 1..3 to match the vertical scale of axis 0 (host axis)
    # -------------------------------
    # We'll keep the first column as is, and transform the others
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    for i in range(1, num_dims):
        # Transform the data so that axis i aligns vertically with axis 0's scale
        zs[:, i] = (ys[:, i] - ymins[i]) / dys[i] * dys[0] + ymins[0]

    # -------------------------------
    # 3. Create the main figure and host axis
    # -------------------------------
    fig, host = plt.subplots(figsize=(10, 6))
    # Create twinned axes for the remaining dimensions
    axes = [host] + [host.twinx() for _ in range(num_dims - 1)]

    # Set each axis's y-limits to its original (unpadded) range
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax is not host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            # Evenly position the right spine along the host axis
            ax.spines["right"].set_position(("axes", i / (num_dims - 1)))

    # -------------------------------
    # 4. Configure the host x-axis
    # -------------------------------
    host.set_xlim(0, num_dims - 1)
    host.set_xticks(range(num_dims))
    host.set_xticklabels(axis_labels, fontsize=12)
    host.tick_params(axis='x', which='major', pad=7)
    host.xaxis.tick_top()
    host.spines['right'].set_visible(False)
    host.set_title(title, fontsize=14)

    # Adjust offset text for the third axis in a 4-axis parallel-coordinates setup
    axes[2].ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
    axes[2].yaxis.set_offset_position('left')  # 'left' also possible

    # Grab the offset text object for 2nd axis
    offset_text = axes[2].yaxis.get_offset_text()

    # Adjust (x, y) to position the text; (1, 0) is the top-right corner of the plot
    # (0, 0) is the top-left corner of the plot
    offset_text.set_position((0.70, 0))

    # -------------------------------
    # 5. Create a colormap for LCST values
    # -------------------------------
    lcst_min, lcst_max = lcst_values.min(), lcst_values.max()
    norm_lcst = mcolors.Normalize(vmin=lcst_min, vmax=lcst_max)
    cmap = cm.seismic  # Choose colour map here

    def get_color(val):
        return cmap(norm_lcst(val))

    # -------------------------------
    # 6. Plot each sample as a smooth Bezier curve
    # -------------------------------
    # For each sample, create control vertices for the Bezier curve.
    # We follow the method: at each axis there is a control vertex, plus one each in-between.
    # Total control points = num_dims * 3 - 2. (First and last axes get 2, others get 3 each.)
    for j in range(num_samples):
        # Generate x-coordinates: linearly spaced between 0 and num_dims-1.
        # Total points: num_dims*3 - 2
        bezier_x = np.linspace(0, num_dims - 1, num=num_dims * 3 - 2, endpoint=True)
        # For y-coordinates, repeat each transformed value 3 times (except first and last only twice)
        bezier_y = np.repeat(zs[j, :], 3)[1:-1]
        verts = list(zip(bezier_x, bezier_y))
        codes = [Path.MOVETO] + [Path.CURVE4] * (len(verts) - 1)
        path = Path(verts, codes)
        patch = PathPatch(path, facecolor='none', lw=1, edgecolor=get_color(lcst_values[j]))
        host.add_patch(patch)

    # -------------------------------
    # 7. Final layout and saving
    # -------------------------------
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300)


def plot_parallel_coords_2(*independent_vars, lcst_values, title, labels, cmap="seismic", save_name=None):
    """
    Plots parallel coordinates with dynamic axes based on input variables.

    Args:
        *independent_vars: Variable number of array-like independent variables
        lcst_values (array-like): Dependent variable (LCST values)
        title (str): Plot title
        labels (list): Labels for each independent variable
        save_name (str, optional): Output filename
    """
    # Input validation
    if not independent_vars:
        raise ValueError("At least one independent variable required")
    if len(labels) != len(independent_vars):
        raise ValueError("Number of labels must match independent variables")

    # Convert to numpy arrays and validate shapes
    independent_arrays = [np.asarray(var) for var in independent_vars]
    lcst_array = np.asarray(lcst_values)
    n_samples = len(lcst_array)

    for arr in independent_arrays:
        if len(arr) != n_samples:
            raise ValueError("All variables must have same number of samples")

    # Combine data and setup dimensions
    data = np.column_stack(independent_arrays + [lcst_array])
    num_dims = data.shape[1]
    axis_labels = labels + ["LCST"]

    # Normalization with padding
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    mins -= ranges * 0.05
    maxs += ranges * 0.05
    ranges = maxs - mins

    # Transform data to first axis' scale
    transformed = np.zeros_like(data)
    transformed[:, 0] = data[:, 0]
    for i in range(1, num_dims):
        transformed[:, i] = (data[:, i] - mins[i]) / ranges[i] * ranges[0] + mins[0]

    # Create figure and axes
    fig, host = plt.subplots(figsize=(10, 6))
    axes = [host] + [host.twinx() for _ in range(num_dims - 1)]

    # Axis positioning and styling
    for i, ax in enumerate(axes):
        ax.set_ylim(mins[i], maxs[i])
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)

        if ax != host:
            ax.spines.left.set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines.right.set_position(("axes", i / (num_dims - 1)))

    # Host axis configuration
    host.set_xlim(0, num_dims - 1)
    host.set_xticks(range(num_dims))
    host.set_xticklabels(axis_labels, fontsize=12)
    host.tick_params(axis='x', which='major', pad=7)
    host.xaxis.tick_top()
    host.spines.right.set_visible(False)
    host.set_title(title, fontsize=14, pad=20)

    # Adjust offset text for the third axis in a 4-axis parallel-coordinates setup
    axes[2].ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
    axes[2].yaxis.set_offset_position('left')  # 'left' also possible

    # Grab the offset text object for 2nd axis
    offset_text = axes[2].yaxis.get_offset_text()

    # Adjust (x, y) to position the text; (1, 0) is the top-right corner of the plot
    # (0, 0) is the top-left corner of the plot
    offset_text.set_position((0.70, 0))

    # Get colormap
    cmap = plt.get_cmap(cmap)  # Convert string to colormap
    norm = mcolors.Normalize(vmin=lcst_array.min(), vmax=lcst_array.max())

    # Bezier curve plotting
    for j in range(n_samples):
        bezier_x = np.linspace(0, num_dims - 1, num=num_dims * 3 - 2)
        bezier_y = np.repeat(transformed[j], 3)[1:-1]
        path = Path(np.column_stack([bezier_x, bezier_y]),
                    [Path.MOVETO] + [Path.CURVE4] * (len(bezier_x) - 1))
        host.add_patch(PathPatch(path, facecolor='none', lw=1,
                                 edgecolor=cmap(norm(lcst_array[j]))))

    # # Or linear plot
    # for j in range(n_samples):
    #     path = Path(np.column_stack([range(num_dims), transformed[j]]),
    #                 [Path.MOVETO] + [Path.LINETO] * (num_dims - 1))
    #     host.add_patch(PathPatch(path, facecolor='none', lw=1,
    #                              edgecolor=cmap(norm(lcst_array[j]))))

    # Final layout
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()


def plot_3D_surface(x, y, z, lcst_values, title, labels, cmap, save_name):
    """
    Generates a 3D surface plot with the surface color-mapped to LCST values.

    Parameters:
    x (array-like): 1D array of Polymer concentrations.
    y (array-like): 1D array of NaCl concentrations.
    z (array-like): 1D array of HCl concentrations.
    lcst_values (array-like): 1D array of LCST values corresponding to each data point.
    title (str): Title of the plot.
    labels (list of str): Labels for the x, y, and z axes.
    cmap (str or Colormap): Colormap for the LCST values.
    save_name (str): File path to save the plot.
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a triangulation from x and y coordinates
    triang = mtri.Triangulation(x, y)

    # Normalize LCST values for coloring
    norm = plt.Normalize(vmin=min(lcst_values), vmax=max(lcst_values))
    colors = plt.cm.get_cmap(cmap)(norm(lcst_values))

    # Plot the surface (initially without colors)
    surf = ax.plot_trisurf(triang, z, cmap=cmap, edgecolor='none')

    # Set the correct colors for the faces
    surf.set_facecolors(colors)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('LCST')

    # Set axis labels and title
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    plt.title(title)

    # Save the plot
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()


def first_below_threshold_index(row):
    # Iterate over the row with index and return the first index where value < threshold
    for idx, value in enumerate(row):
        if value < 50:
            return idx
    return None  # Return None if no values are below the threshold in the row


def fit_sigmoidal_and_plot(temperature, transmittance, out_path):
    def sigmoidal(x, A1, A2, x0, dx):
        return A2 + (A1 - A2) / (1 + np.exp((x - x0) / dx))

    # global inflection_temps
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
    ax.set_title("Turbidity Curves for Predicted LCST of 32.5 °C", fontsize=16, pad=15)

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


def train_xgboost_model(X_train, y_train, generations=10, population_size=75, scoring="r2",
                        random_state=42, cv=5, n_jobs=-1):
    """
    Train and optimize hyperparameters for an XGBoost model using randomized search.

    Args:
        X_train (array-like): Training data for features
        y_train (array-like): Training data for target variable
        generations (int, optional): Number of generations (used to calculate n_iter). Default 10
        population_size (int, optional): Population size per generation (used to calculate n_iter). Default 75
        scoring (str, optional): Scoring metric to optimize. Default "r2"
        random_state (int, optional): Seed for reproducibility. Default 42
        cv (int, optional): Number of cross-validation folds. Default 5
        n_jobs (int, optional): Number of parallel jobs. Default -1 (all cores)

    Returns:
        RandomizedSearchCV: Optimized search object containing best model
    """
    try:
        # Calculate total iterations based on TPOT-like parameters
        n_iter = generations * population_size

        # Define hyperparameter distributions
        param_dist = {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),  # Range 0.6-1.0
            'colsample_bytree': uniform(0.6, 0.4),  # Range 0.6-1.0
            'gamma': uniform(0, 0.5),  # Regularization
            'reg_alpha': uniform(0, 1),  # L1 regularization
            'reg_lambda': uniform(0, 1)  # L2 regularization
        }

        # Initialize base model
        xgb = XGBRegressor(random_state=random_state, n_jobs=n_jobs)

        # Set up randomized search
        search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            verbose=2,
            random_state=random_state,
            n_jobs=n_jobs,
            return_train_score=True
        )

        # Perform hyperparameter optimization
        search.fit(X_train, y_train)

        logging.info(f"Best parameters: {search.best_params_}")
        logging.info(f"Best {scoring} score: {search.best_score_:.3f}")

        return search

    except Exception as e:
        logging.error(f"An error occurred during model training: {str(e)}")
        return None


def train_svr_model(X_train, y_train, generations=2, population_size=75, scoring="r2",
                    random_state=42, cv=5, n_jobs=-1):
    """
    Train and optimize hyperparameters for a Support Vector Regression (SVR) model using randomized search.

    Args:
        X_train (array-like): Training data for features
        y_train (array-like): Training data for target variable
        generations (int, optional): Number of generations (used to calculate n_iter). Default 10
        population_size (int, optional): Population size per generation (used to calculate n_iter). Default 75
        scoring (str, optional): Scoring metric to optimize. Default "r2"
        random_state (int, optional): Seed for reproducibility. Default 42
        cv (int, optional): Number of cross-validation folds. Default 5
        n_jobs (int, optional): Number of parallel jobs. Default -1 (all cores)

    Returns:
        RandomizedSearchCV: Optimized search object containing best model
    """
    try:
        # Calculate total iterations based on TPOT-like parameters
        n_iter = generations * population_size

        # Define hyperparameter distributions
        param_dist = {
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'C': reciprocal(1e-3, 1e3),  # Log-uniform distribution for C
            'gamma': reciprocal(1e-5, 1e3),  # Log-uniform distribution for gamma
            'epsilon': uniform(0.01, 0.5),  # Uniform distribution between 0.01 and 0.51
            'degree': randint(2, 5),  # Integer values 2, 3, 4
            'coef0': uniform(-1, 2)  # Uniform distribution between -1 and 1
        }

        # Initialize base model
        svr = SVR()

        # Set up randomized search
        search = RandomizedSearchCV(
            estimator=svr,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            verbose=2,
            random_state=random_state,
            n_jobs=n_jobs,
            return_train_score=True
        )

        # Perform hyperparameter optimization
        search.fit(X_train, y_train)

        logging.info(f"Best parameters: {search.best_params_}")
        logging.info(f"Best {scoring} score: {search.best_score_:.3f}")

        return search

    except Exception as e:
        logging.error(f"An error occurred during model training: {str(e)}")
        return None


def train_NN_model(X_train, y_train):
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import loguniform
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np

    # Pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Add interactions/squares
        ('mlp', MLPRegressor(early_stopping=True, random_state=42)),
    ])

    # Example check for input scaling
    print("Feature means (before scaling):", X_train.mean(axis=0))
    print("Target mean (before scaling):", y_train.mean())

    param_dist = {
        # Smaller networks to reduce overfitting risk
        'mlp__hidden_layer_sizes': [(30, 20), (40, 30), (20, 10), (50, 30, 10)],
        # 'mlp__hidden_layer_sizes': (100, 50),
        'mlp__activation': ['tanh', 'relu'],
        'mlp__solver': ['lbfgs'],  # Good for small dataset
        'mlp__alpha': loguniform(1e-6, 1e-3),  # Wider regularization range
        'mlp__learning_rate_init': loguniform(1e-5, 1e-2),  # Broader learning rates
        'mlp__max_iter': [100000],  # Extreme patience for lbfgs
        'mlp__tol': [1e-7],  # Stricter convergence criteria
        'mlp__n_iter_no_change': [50],  # Longer waiting period
    }

    # Randomized search with fewer iterations
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)
    return search


def train_NN_model_concs(X_train, y_train):
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import loguniform
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np

    # Pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Add interactions/squares
        ('mlp', MLPRegressor(early_stopping=True, random_state=42)),
    ])

    # Example check for input scaling
    print("Feature means (before scaling):", X_train.mean(axis=0))
    print("Target mean (before scaling):", y_train.mean())

    param_dist = {
        # Smaller networks to reduce overfitting risk
        # 'mlp__hidden_layer_sizes': [(30, 20), (40, 30), (20, 10), (50, 30, 10)],
        'mlp__hidden_layer_sizes': (100, 50),
        'mlp__activation': ['tanh', 'relu'],
        'mlp__solver': ['lbfgs'],  # Good for small dataset
        'mlp__alpha': loguniform(1e-6, 1e-3),  # Wider regularization range
        'mlp__learning_rate_init': loguniform(1e-5, 1e-2),  # Broader learning rates
        'mlp__max_iter': [10000],  # Extreme patience for lbfgs
        'mlp__tol': [1e-6],  # Stricter convergence criteria
        'mlp__n_iter_no_change': [25],  # Longer waiting period
    }

    # Randomized search with fewer iterations
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=25,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)
    return search


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
        log_msg(f"An error occurred while training the model: {e}")
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
    plt.savefig((out_path + "_predicted_vs_actual.png"), dpi=300)
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
        PartialDependenceDisplay.from_estimator(model, X_train, features=[0, 1, 2], ax=ax)
        ax.set_title("Partial Dependence Plots", fontsize=16)
        plt.savefig(os.path.join(insight_out_path, "partial_dependence.png"), dpi=300)
        plt.close()
    except Exception as e:
        log_msg(f"Partial Dependence Error: {e}")

    # Permutation Importance
    try:
        result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
        sorted_idx = result.importances_mean.argsort()[::-1]

        plt.figure(figsize=(12, 8))
        plt.bar(range(len(sorted_idx)), result.importances_mean[sorted_idx], color="#41424C")
        plt.xticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.title("Feature Importance via Permutation", fontsize=16)
        plt.savefig(os.path.join(insight_out_path, "permutation_importance.png"), dpi=300)
        plt.close()
    except Exception as e:
        log_msg(f"Permutation Importance Error: {e}")

    # Interaction Contour Plots for each feature pair
    try:
        feature_pairs = list(combinations(enumerate(feature_names), 2))
    except Exception as e:
        log_msg(f"Error generating feature pairs: {e}")
        feature_pairs = []

    for (var1_index, var1_name), (var2_index, var2_name) in feature_pairs:
        try:
            # Indices of features to fix (all except var1 and var2)
            fixed_indices = [i for i in range(len(feature_names)) if i not in (var1_index, var2_index)]
            fixed_values = {i: np.mean(X_train[:, i]) for i in fixed_indices}

            # Create ranges for the varying features
            var1_range = np.linspace(X_train[:, var1_index].min(), X_train[:, var1_index].max(), 50)
            var2_range = np.linspace(X_train[:, var2_index].min(), X_train[:, var2_index].max(), 50)

            # Generate meshgrid
            var1_mesh, var2_mesh = np.meshgrid(var1_range, var2_range)

            # Create grid points with fixed features at their mean
            grid_points = np.tile(np.mean(X_train, axis=0), (var1_mesh.size, 1))
            grid_points[:, var1_index] = var1_mesh.ravel()
            grid_points[:, var2_index] = var2_mesh.ravel()

            # Predict
            predictions = model.predict(grid_points).reshape(var1_mesh.shape)

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            contour = ax.contourf(var1_mesh, var2_mesh, predictions, cmap='coolwarm', levels=20)
            cbar = plt.colorbar(contour)
            cbar.set_label("Predicted LCST (°C)", fontsize=12)

            ax.set_xlabel(var1_name, fontsize=14, labelpad=10)
            ax.set_ylabel(var2_name, fontsize=14, labelpad=10)

            # Generate title with fixed features if any
            fixed_vars = [f"{feature_names[i]}={fixed_values[i]:.3f}" for i in fixed_indices]
            title = f"LCST vs {var1_name} and {var2_name}"
            if fixed_vars:
                title += f"\n(Fixed: {', '.join(fixed_vars)})"
            ax.set_title(title, fontsize=16, pad=15)

            plt.tight_layout()
            plt.savefig(
                os.path.join(insight_out_path, f"contour_{var1_name}_vs_{var2_name}.png"),
                dpi=300,
            )
            plt.close()
        except Exception as e:
            log_msg(f"Contour Plot Error ({var1_name} vs {var2_name}): {e}")


def optimize_concentration(model, target_lcst, bounds=(0, 10), fixed_values=[0.142 / 100, 1.25 / 100000]):
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


def run():
    # -------------------------------
    # Configuration: Define file paths and dataset names
    # -------------------------------
    out_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\LCST\Copolymers\FR-NIPAM-DMAEMA-4-NaOH_10-2"

    params = {
        "json_files": [
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\FR-NIPAM-DMAEMA-1\\temperature_data_2025-03-13 15_28_28.json",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\FR-NIPAM-DMAEMA-2\\temperature_data_2025-03-14 11_51_25.json",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\FR-NIPAM-DMAEMA-3\\temperature_data_2025-04-23 16_11_17.json",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\FR-NIPAM-DMAEMA-4\\temperature_data_2025-04-24 10_47_17.json",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\FR-NIPAM-DMAEMA-5\\temperature_data_2025-04-29 13_45_05.json",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\23-Jan NIPAM + 0.01 M NaCl + 10e-4 M HCl\\temperature_data_2025-01-23 16_38_52.json",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\RAFT-NIPAM-DMAEMA-1\\temperature_data_2025-02-18 16_05_40.json",
            r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\LCST\Copolymers\FR-NIPAM-DMAEMA-4-NaOH_10-2\temperature_data_2025-05-06 16_28_26.json",
        ],
        "volumes_files": [
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\Duplicated_Volumes.csv",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\Duplicated_Volumes.csv",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\Duplicated_Volumes.csv",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\Duplicated_Volumes.csv",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\Duplicated_Volumes.csv",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\23-Jan NIPAM + 0.01 M NaCl + 10e-4 M HCl\\Duplicated_Volumes_1.csv",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\Duplicated_Volumes.csv",
            r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\Duplicated_Volumes.csv",
        ],
        "spectra_folders": [
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\FR-NIPAM-DMAEMA-1\\abs_spectra",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\FR-NIPAM-DMAEMA-2\\abs_spectra",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\FR-NIPAM-DMAEMA-3\\abs_spectra",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\FR-NIPAM-DMAEMA-4\\abs_spectra",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\FR-NIPAM-DMAEMA-5\\abs_spectra",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\23-Jan NIPAM + 0.01 M NaCl + 10e-4 M HCl\\abs spectra",
            # r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\LCST\\Copolymers\\RAFT-NIPAM-DMAEMA-1\\abs_spectra",
            r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\LCST\Copolymers\FR-NIPAM-DMAEMA-4-NaOH_10-2\abs_spectra",
        ],
        "names": [
            # "PNIPAM-DMAEMA-FR-1",
            # "PNIPAM-DMAEMA-FR-2",
            # "PNIPAM-DMAEMA-FR-3",
            # "PNIPAM-DMAEMA-FR-4",
            # "PNIPAM-DMAEMA-FR-5",
            # "DoPAT-PNIPAM-DP60",
            # "CDTP-PNIPAM-DP165",
            "FR-NIPAM-DMAEMA-4-NaOH_10-2",
        ]
    }

    # Initialize dictionary to store processed data for each dataset
    data = {}

    # -------------------------------
    # Data Loading and Initial Processing Loop
    # -------------------------------
    # Check if all lists in the params dictionary have the same length
    f = len(params[next(iter(params))])
    if all(len(x) == f for x in params.values()):
        pass
    else:
        raise ValueError("All items in params dictionary should be of the same length.")

    # Loop through each dataset defined in the params dictionary
    for i in range(f):
        try:
            # Attempt to prepare data using specified wavelength range (600nm)
            log_msg(f"Processing dataset: {params['names'][i]} with wavelength 600nm")
            data_objects = prepare_all_data(
                json_path=params["json_files"][i],
                volumes_csv_path=params["volumes_files"][i],
                spectra_folder_path=params["spectra_folders"][i],
                out_path=out_path,
                idx_start=None,
                idx_end=None,
                wavelength_start=600,
                wavelength_end=600,
            )
        except Exception as e:
            # If the first attempt fails, log the error and retry with a broader wavelength range
            log_msg(f"Error processing {params['names'][i]} at 600nm: {e}")
            log_msg("Re-attempting data read with full wavelength range (220-1000nm)...")

            # Prepare data again with the full wavelength range
            data_objects = prepare_all_data(
                json_path=params["json_files"][i],
                volumes_csv_path=params["volumes_files"][i],
                spectra_folder_path=params["spectra_folders"][i],
                out_path=out_path,
                idx_start=None,
                idx_end=None,
                wavelength_start=220,
                wavelength_end=1000,
            )

        # Store the processed data objects in the dictionary using the dataset name as the key
        data[params["names"][i]] = data_objects

    print("Data loading and initial processing complete. Data dictionary keys:", data.keys())

    # -------------------------------
    # Dataset-Specific Processing and Analysis Loop
    # -------------------------------
    # Collect temperature_values_filtered for all datasets for the combined box plot
    all_temperature_values_filtered = []

    # Loop through each processed dataset stored in the 'data' dictionary
    for name, data_objects in data.items():
        log_msg(f"--- Starting analysis for dataset: {name} ---")
        # -------------------------------
        # Unpack Data Objects for Current Dataset
        # -------------------------------
        stacked_transmittance_df = data_objects['stacked_transmittance_df']
        concentrations = data_objects['concentrations']
        # averages = data_objects['averages'] # Original averages from transmittance calculation (commented out)
        measurement_times = data_objects['measurement_times']
        temps1_plotting = data_objects['temps1_plotting']
        volumes_df = data_objects['volumes_df']
        inflection_temps = data_objects['inflection_temps']
        dx_values = data_objects['dx_values']

        # Convert inflection temperatures and dx values to NumPy arrays
        inflection_temps = np.array(inflection_temps)
        dx_values = np.array(dx_values)

        # -------------------------------
        # Outlier detection
        # -------------------------------
        outlier_method = 'iqr'  # or 'zscore' or None
        zscore_threshold = 3.0
        iqr_multiplier = 1.5

        data_no_nan = inflection_temps[~np.isnan(inflection_temps)]

        if outlier_method == 'zscore':
            zscores = np.abs((data_no_nan - np.mean(data_no_nan)) / np.std(data_no_nan))
            outlier_mask = zscores > zscore_threshold
            outlier_indices = np.where(~np.isnan(inflection_temps))[0][outlier_mask]
        elif outlier_method == 'iqr':
            q1 = np.percentile(data_no_nan, 25)
            q3 = np.percentile(data_no_nan, 75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            outlier_indices = np.where(
                (inflection_temps < lower_bound) | (inflection_temps > upper_bound)
            )[0]
        else:
            outlier_indices = []

        # Replace outliers with np.nan
        # inflection_temps[outlier_indices] = np.nan

        # -------------------------------
        # Calculate Statistics from Inflection Temperatures
        # -------------------------------

        # Reshape into pairs (assuming duplicate measurements for each condition)
        pairs = inflection_temps.reshape(-1, 2)
        dx_pairs = dx_values.reshape(-1, 2)

        # Compute mean, standard deviation, and RSD for each pair
        averages = pairs.mean(axis=1)
        std_devs = pairs.std(axis=1, ddof=1)  # Use sample standard deviation (ddof=1)
        rsd = (std_devs / averages) * 100

        # Compile statistics into a DataFrame and save to CSV
        df_stats = pd.DataFrame({'Average': averages, 'Standard Deviation': std_devs, '%RSD': rsd})
        stats_filename = os.path.join(out_path, f"stats_{name}.csv")
        df_stats.to_csv(stats_filename, index=False)
        log_msg(f"Statistics saved to: {stats_filename}")

        # -------------------------------
        # Plotting: Combined Transmittance vs. Temperature/Time
        # -------------------------------
        # log_msg("Generating combined transmittance plots...")
        # # Plot individual transmittance curves vs. temperature on one graph
        # plot_transmittance(
        #     plot_type="individual",
        #     x_data=temps1_plotting,
        #     y_data=stacked_transmittance_df.iloc[:, :],
        #     labels=[f"{round(volumes_df.iloc[i, 0] * (10 / 300), 2)} mg/mL" for i in range(len(stacked_transmittance_df))],
        #     title=f'Percent Transmittance at 600 nm of {name} vs. Temperature',
        #     xlabel="Temperature (°C)",
        #     ylabel='Average Transmittance (%)',
        #     save_name=f"trans_versus_temp_individual_{name}.png",
        #     out_path=out_path
        # )
        #
        # # Plot individual transmittances over time (calculated relative to the first measurement)
        # try:
        #     times = [time_difference(measurement_times[0], t) for t in measurement_times]
        #     plot_transmittance(
        #         plot_type="individual",
        #         x_data=times,
        #         y_data=stacked_transmittance_df,
        #         labels=[f"{round(volumes_df.iloc[i, 0] * (10 / 300), 2)} mg/mL" for i in range(len(stacked_transmittance_df))],
        #         title=f'Percent Transmittance at 600 nm of {name} vs. Time',
        #         xlabel="Time (Seconds)",
        #         ylabel='Average Transmittance (%)',
        #         save_name=f"trans_versus_time_individual_{name}.png",
        #         out_path=out_path
        #     )
        # except Exception as e:
        #     log_msg(f"Could not plot transmittance vs. time for {name}: {e}")
        #
        # # Plot averaged transmittances (from paired measurements) over temperature with error bars
        # plot_transmittance(
        #     plot_type="averaged",
        #     x_data=temps1_plotting[:len(averages)], # Match x_data length to calculated averages
        #     y_data=averages,
        #     y_err=std_devs,
        #     title=f'Percent Transmittance at 600 nm of {name} vs. Temperature',
        #     xlabel="Temperature (°C)",
        #     ylabel='Average Transmittance (%)',
        #     save_name=f"trans_versus_temp_averaged_{name}.png",
        #     out_path=out_path
        # )

        # -------------------------------
        # Plotting: Individual Spectra
        # -------------------------------
        # Transmittance versus temperature
        trans_path = os.path.join(out_path, 'individual_spectra', 'trans_temp')
        temp_path = os.path.join(out_path, 'individual_spectra', 'trans_time')
        os.makedirs(trans_path, exist_ok=True)
        os.makedirs(temp_path, exist_ok=True)

        ind = False  # Set to false if individual plots are desired
        if ind:
            log_msg("Generating individual transmittance plots...")
            # Iterate through each sample (row in stacked_transmittance_df)
            for i in range(stacked_transmittance_df.shape[0]):
                try:
                    # Prepare data for the current sample
                    x_data = temps1_plotting
                    y_data = stacked_transmittance_df.iloc[i, :]
                    # Calculate concentration for labelling
                    concentration_val = volumes_df.iloc[i, 0] * (10 / 300)
                    concentration_label = f"{round(concentration_val, 3)} mg/mL"  # Increased precision

                    # --- Transmittance vs Temperature Plot ---
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(x_data, y_data, 'o-', markersize=4, linewidth=1.0)  # Slightly smaller markers/lines

                    ax.set_xlabel("Temperature (°C)", fontsize=12, labelpad=8)
                    ax.set_ylabel("Transmittance (%)", fontsize=12, labelpad=8)
                    plot_title = f'{name} - Well {i + 1} - Conc: {concentration_label}'
                    ax.set_title(plot_title, fontsize=14, pad=12)

                    # Add text label with all component concentrations
                    current_concs = concentrations[i]
                    label_text = f"Comp 1: {current_concs[0]:.3f}\n" \
                                 f"Comp 2: {current_concs[1]:.4f}\n" \
                                 f"Comp 3: {current_concs[2]:.5f}"
                    ax.text(0.05, 0.05, label_text, transform=ax.transAxes, fontsize=9,
                            verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

                    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.tick_params(axis='both', which='major', labelsize=10, width=1.5, length=10)
                    ax.tick_params(axis='both', which='minor', labelsize=10, width=1.5, length=5)
                    # Bold tick labels (manual font weight)
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label1.set_fontweight('bold')
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label1.set_fontweight('bold')
                    ax.grid(False)  # Consistent with other plots

                    plt.tight_layout()

                    # Create a safe filename (replace potential problematic characters if needed)
                    safe_conc_label = str(round(concentration_val, 3)).replace('.', 'p')  # e.g., 5.2 -> 5p2
                    save_filename = os.path.join(trans_path,
                                                 f"trans_temp_{name}_well_{i + 1}_conc_{safe_conc_label}.png")
                    plt.savefig(save_filename, dpi=300)
                    plt.close(fig)  # Close the figure to free memory

                    # --- Transmittance vs Time Plot ---
                    times = [time_difference(measurement_times[0], t) for t in measurement_times]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(times, y_data, 'o-', markersize=4, linewidth=1.0)  # Slightly smaller markers/lines

                    ax.set_xlabel("Time (mins)", fontsize=12, labelpad=8)
                    ax.set_ylabel("Transmittance (%)", fontsize=12, labelpad=8)
                    # Use the same title format as the temp plot
                    plot_title = f'{name} - Well {i + 1} - Conc: {concentration_label}'
                    ax.set_title(plot_title, fontsize=14, pad=12)

                    # Add text label with all component concentrations (same as above)
                    ax.text(0.05, 0.05, label_text, transform=ax.transAxes, fontsize=9,
                            verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

                    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.tick_params(axis='both', which='major', labelsize=10, width=1.5, length=10)
                    ax.tick_params(axis='both', which='minor', labelsize=10, width=1.5, length=5)
                    # Bold tick labels (manual font weight)
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label1.set_fontweight('bold')
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label1.set_fontweight('bold')
                    ax.grid(False)  # Consistent with other plots

                    plt.tight_layout()

                    # Create a safe filename (replace potential problematic characters if needed)
                    safe_conc_label = str(round(concentration_val, 3)).replace('.', 'p')  # e.g., 5.2 -> 5p2
                    save_filename = os.path.join(temp_path,
                                                 f"trans_time_{name}_well_{i + 1}_conc_{safe_conc_label}.png")
                    plt.savefig(save_filename, dpi=300)
                    plt.close(fig)  # Close the figure to free memory

                except IndexError:
                    log_msg(
                        f"Skipping plot for index {i} due to IndexError (likely mismatch in volumes_df or stacked_transmittance_df dimensions).")
                except Exception as e:
                    log_msg(f"Error plotting individual spectrum for well {i + 1} in dataset {name}: {e}")

        # -------------------------------
        # Sigmoidal Fitting and Temperature Extraction
        # -------------------------------
        log_msg("Fitting sigmoidal curves to transmittance data...")
        temperature = temps1_plotting[:len(temps1_plotting) // 2]  # Assuming heating cycle is first half
        transmittance = stacked_transmittance_df
        fit_sigmoidal_and_plot(temperature, transmittance,
                               out_path)  # Note: This re-calculates inflection temps internally

        # Filter out any None values from the inflection temperatures calculated earlier
        temperature_values_filtered = np.array([temp for temp in inflection_temps[:] if temp is not None])
        all_temperature_values_filtered.append(temperature_values_filtered)  # Collect for combined plot later

        log_msg(f"Filtered inflection temperatures shape: {temperature_values_filtered.shape}")
        log_msg(f"Concentrations shape: {concentrations.shape}")

        # -------------------------------
        # Data Visualization: Heatmap and Parallel Coordinates
        # -------------------------------
        log_msg("Generating heatmap and parallel coordinates plots...")
        # Create a heatmap of the filtered LCST values (inflection temps)
        create_heatmap(
            temperature_values_filtered,
            figsize=(9, 6),
            filename=os.path.join(out_path, f"heatmap_seismic_{name}.png"),
            cmap="seismic"
        )

        plot_parallel_coords_2(
            concentrations[4:, 0],
            concentrations[4:, 1],
            concentrations[4:, 2],
            lcst_values=temperature_values_filtered[4:],
            title=f"Parallel Coordinates Plot for LCST vs. Polymer and NaCl Concentrations - {name}",
            labels=[f"[{name}]", "[NaCl]", "[NaOH]"],
            cmap="seismic",
            save_name=os.path.join(out_path, f"parallel_coords_seismic_{name}.png")
        )

        # -------------------------------
        # Machine Learning Model Training (Neural Network)
        # -------------------------------
        log_msg("Preparing data and training NN model...")
        # Define feature names based on the current dataset name
        FEATURE_NAMES = [f"{name}", "NaCl", "NaOH"]

        # Prepare data, excluding the first 4 samples (often controls or blanks)
        # Ensure concentration and temperature arrays are aligned and match in length after filtering
        valid_indices = np.arange(len(temperature_values_filtered)) >= 4
        X_data_ml = concentrations[valid_indices, :]
        y_data_ml = temperature_values_filtered[valid_indices]

        if len(X_data_ml) != len(y_data_ml):
            log_msg(
                f"Warning: Mismatch between features ({len(X_data_ml)}) and target ({len(y_data_ml)}) for ML for dataset {name}. Skipping ML.")
            continue  # Skip ML for this dataset if lengths don't match

        if len(y_data_ml) < 5:  # Need enough data for train/test split and CV
            log_msg(f"Warning: Not enough data points ({len(y_data_ml)}) for ML for dataset {name}. Skipping ML.")
            continue

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = prepare_data(X_data_ml, y_data_ml)

        # Train the Neural Network model using Randomized Search for hyperparameter tuning
        result = train_NN_model(X_train, y_train)
        if result is None:
            log_msg(f"NN Model training failed for dataset {name}. Skipping evaluation and insights.")
            continue
        model = result.best_estimator_
        log_msg(f"NN Model trained successfully for {name}.")

        # -------------------------------
        # Model Evaluation and Insights Generation
        # -------------------------------
        log_msg("Evaluating model and generating insights...")
        # Evaluate the trained model on the test set
        evaluate_model(model, X_test, y_test, os.path.join(out_path, f"model_evaluation_{name}"))
        # Generate insights plots (Partial Dependence, Permutation Importance, Contour Plots)
        generate_model_insights(
            model, X_train, X_test, y_test,
            os.path.join(out_path, f"insights_{name}"), FEATURE_NAMES
        )

        # -------------------------------
        # Data Distribution Analysis: Histogram and Moments
        # -------------------------------
        log_msg("Generating histogram and calculating moments for inflection temperatures...")
        print(f"\n--- Generating Histogram and Moments for {name} ---")
        # Use only the data points used for ML (excluding first 4) for consistency
        temps_for_hist = y_data_ml
        print(f"Number of observed temperatures for histogram: {len(temps_for_hist)}")

        plt.figure(figsize=(10, 6))
        # Use 'auto' bins or specify a number, ensure data is not empty
        if len(temps_for_hist) > 0:
            counts, bins, patches = plt.hist(temps_for_hist, bins=12, density=False, alpha=0.75,
                                             edgecolor='black')
            plt.title(f'Distribution of Observed Inflection Temperatures for {name} (Samples 5+)')
            plt.xlabel("Inflection Temperature (°C)")
            plt.ylabel('Frequency (Count)')
            plt.grid(False)
            plt.tight_layout()

            # Save the histogram
            hist_save_name = f"histogram_inflection_temps_{name}.png"
            hist_filename = os.path.join(out_path, hist_save_name)
            try:
                plt.savefig(hist_filename, dpi=300)
                print(f"Histogram saved to: {hist_filename}")
            except Exception as e:
                print(f"Error saving histogram {hist_filename}: {e}")
            plt.close()  # Close the plot

            # Calculate the first four moments using the same filtered data
            mean_temp = np.mean(temps_for_hist)
            variance_temp = np.var(temps_for_hist, ddof=0)  # Population variance
            skewness_temp = stats.skew(temps_for_hist)
            kurtosis_temp = stats.kurtosis(temps_for_hist, fisher=True)  # Excess kurtosis

            print(f"Moments for {name} (Samples 5+):")
            print(f"  1. Mean: {mean_temp:.4f}")
            print(f"  2. Variance: {variance_temp:.4f}")
            print(f"  3. Skewness: {skewness_temp:.4f}")
            print(f"  4. Kurtosis (Excess): {kurtosis_temp:.4f}")
        else:
            print(f"No valid temperature data to generate histogram or moments for {name}.")

    # -------------------------------
    # Combined Analysis Across All Datasets
    # -------------------------------
    log_msg("Generating combined box plot for all datasets...")
    # Create a combined box plot comparing LCST distributions across all datasets
    create_boxplot(
        all_temperature_values_filtered,  # List containing filtered temp arrays from each dataset
        names=params["names"],  # Use dataset names as labels
        figsize=(8, 6),  # Adjust size as needed
        filename=os.path.join(out_path, "boxplot_all_datasets.png"),
        label_rotation=30  # Rotate labels slightly for readability
    )
    log_msg("Combined box plot saved.")

    # -------------------------------
    # Optional: 3D Plotting (Commented Out)
    # -------------------------------
    # plot_3D_surface(
    #     concentrations[4:, 0],
    #     concentrations[4:, 1],
    #     concentrations[4:, 2],
    #     lcst_values=model.predict(concentrations[4:, :]),
    #     title="3D Surface Plot for LCST versus Polymer, NaCl, and HCl Concentrations",
    #     labels=["[Polymer]", "[NaCl]", "[HCl]"],
    #     cmap="seismic",
    #     save_name=out_path+rf"\3D_plot_seismic.png")

    # # Optimization and solution generation
    # bounds = (X_train[:, 0].min(), X_train[:, 0].max())
    # target_LCST = 32.5  # Target temperature
    # optimal = optimize_concentration(model, target_LCST, bounds=(0, 10), fixed_values=[0.142/100, 2/100000])
    # if optimal:
    #     generate_volumes_csv(optimal, out_path)


def run_conc_modelling():
    # -------------------------------
    # Concentration Modelling Setup (Separate Function - Potentially Obsolete/Experimental)
    # -------------------------------
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error

    out_path = r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\DOE + Monomer + Polymer Mixtures\\Multivariable Experiments\\01-Nov three factor - RAFT-PS"
    volumes = r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\DOE + Monomer + Polymer Mixtures\\Multivariable Experiments\\Duplicated_Volumes.csv"
    plate = r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\DOE + Monomer + Polymer Mixtures\\Multivariable Experiments\\04-Nov three factor - toluene\\abs spectra\\241104_1530.csv"
    data = r"C:\\Users\\Lachlan Alexander\\Desktop\\Uni\\2024 - Honours\\Experiments\\DOE + Monomer + Polymer Mixtures\\Multivariable Experiments\\04-Nov three factor - toluene\\abs spectra\\241104_1549.csv"

    data_corrected = separate_subtract_and_recombine(load_data_new(data), load_data_new(plate)).to_numpy()

    # Load in volumes
    volumes = load_data(volumes).to_numpy()

    # Correct from volume to concentration
    volumes[:, 0] *= 0.025 / 300
    volumes[:, 1] *= 0.25 / 300
    volumes[:, 2] *= 0.025 / 300

    # Extract features (absorbance spectra) and targets (concentrations)
    X = data_corrected[:, 40:80]  # Absorbance spectra
    y = volumes[:, :3]  # Concentrations of styrene, polystyrene, RAFT-PS

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    model = Ridge()

    # Perform Grid Search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get best model
    model = grid_search.best_estimator_

    # model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    log_msg(r2)

    # Calculate R^2 and MSE for each target variable
    r2_styrene = r2_score(y_test[:, 0], y_pred[:, 0])
    mse_styrene = mean_squared_error(y_test[:, 0], y_pred[:, 0])

    r2_polystyrene = r2_score(y_test[:, 1], y_pred[:, 1])
    mse_polystyrene = mean_squared_error(y_test[:, 1], y_pred[:, 1])

    r2_tol = r2_score(y_test[:, 2], y_pred[:, 2])  # Should match index 2
    mse_tol = mean_squared_error(y_test[:, 2], y_pred[:, 2])  # Should match index 2

    log_msg(
        f"Model: R^2 = {r2_styrene:.4f}/{r2_polystyrene:.4f}/{r2_tol:.4f} - "
        f"MSE: {mse_styrene:}/{mse_polystyrene:}/{mse_tol:}"
    )

    # Create subplots: one row for each model, two columns for styrene and polystyrene
    fig, axes = plt.subplots(1, 3, figsize=(12, 4 * 1))
    fig.suptitle('Model Predictions vs Actual Concentrations', fontsize=16)

    i = 0

    # Styrene (first column of y)
    axes[0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.7)
    axes[0].plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2)
    axes[0].set_xlabel('Actual Styrene Concentration')
    axes[0].set_ylabel('Predicted Styrene Concentration')
    axes[0].set_title(f'{"Ridge"} - Styrene')

    # Add R² and MSE as text annotations
    axes[0].text(0.05, 0.9, f'R² = {r2_styrene:.4f}\nMSE = {mse_styrene:.4f}',
                 transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.5))

    # Polystyrene (second column of y)
    axes[1].scatter(y_test[:, 1], y_pred[:, 1], alpha=0.7)
    axes[1].plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2)
    axes[1].set_xlabel('Actual Polystyrene Concentration')
    axes[1].set_ylabel('Predicted Polystyrene Concentration')
    axes[1].set_title(f'{"Ridge"} - Polystyrene')

    # Add R² and MSE as text annotations
    axes[1].text(0.05, 0.9, f'R² = {r2_polystyrene:.4f}\nMSE = {mse_polystyrene:.4f}',
                 transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.5))

    # Polystyrene (second column of y)
    axes[2].scatter(y_test[:, 2], y_pred[:, 2], alpha=0.7)
    axes[2].plot([y_test[:, 2].min(), y_test[:, 2].max()], [y_test[:, 2].min(), y_test[:, 2].max()], 'k--', lw=2)
    axes[2].set_xlabel('Actual Toluene Concentration')
    axes[2].set_ylabel('Predicted Toluene Concentration')
    axes[2].set_title(f'{"Ridge"} - Toluene')

    # Add R² and MSE as text annotations
    axes[2].text(0.05, 0.9, f'R² = {r2_polystyrene:.4f}\nMSE = {mse_polystyrene:.4f}',
                 transform=axes[2].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.5))

    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path + r"\model_screening_concs_corrected.png")


if __name__ == "__main__":
    # Entry point: Choose which function to run
    # run_conc_modelling() # Uncomment to run the concentration modelling part
    apply_custom_plot_style()  # Homogenise plots
    run()  #Run the main LCST analysis
