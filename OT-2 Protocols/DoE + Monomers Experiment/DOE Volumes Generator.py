import numpy as np
import pyDOE2
import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt
import pandas as pd
import time
import csv


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

# Define constants
num_samples = 46  # the number of unique samples to be measured
total_volume = 300  # final volume in each well
step_size = 20  # minimum step size
num_factors = 2  # number of variables (styrene, polystyrene)


# def generate_lhs_design(num_samples, total_volume, step_size, num_factors):
#     # Generate LHS design
#     lhs = pyDOE2.lhs(num_factors, samples=num_samples)
#
#     # Scale LHS design to the volume constraints
#     scaled_lhs = lhs * (total_volume / num_factors)
#
#     # Round to the nearest step_size and ensure minimum volume constraint
#     scaled_lhs = np.round(scaled_lhs / step_size, 2) * step_size
#     scaled_lhs = np.clip(scaled_lhs, step_size, None)
#
#     # Calculate the third column (solvent) as 300 minus the sum of the first two columns
#     solvent_volumes = total_volume - np.sum(scaled_lhs, axis=1)
#
#     # Combine the first two columns with the new solvent column
#     scaled_lhs = np.column_stack((scaled_lhs, solvent_volumes))
#
#     return scaled_lhs
#
#
# # Generate the LHS design
# scaled_lhs = generate_lhs_design(num_samples, total_volume, step_size, num_factors)
#
# # Verification
# # Check that all rows sum to total_volume
# volume_sums = np.sum(scaled_lhs, axis=1)
#
# while np.all(volume_sums != total_volume) or len(np.unique(scaled_lhs, axis=0)) != num_samples:
#     print("ERROR: Some samples are not unique or do not sum to 300 uL")
#     scaled_lhs = generate_lhs_design(num_samples, total_volume, step_size, num_factors)  # regenerate the design if doesn't pass checks
#     volume_sums = np.sum(scaled_lhs, axis=1)
#
# # Check that all rows sum to total_volume
# volume_sums = np.sum(scaled_lhs, axis=1)
# if np.all(volume_sums == total_volume):
#     print("VERIFIED: All samples have a total volume of 300 units.")
# else:
#     print("ERROR: Some samples do not sum to 300 units.")
#
# # Check that all rows are unique
# if len(np.unique(scaled_lhs, axis=0)) == num_samples:
#     print("VERIFIED: All samples are unique.")
# else:
#     print("ERROR: Some samples are not unique.")
#
# # Print and plot the verified design
# print(scaled_lhs)
# print(volume_sums)
#
# # Visualises the spread of sample points in 2D
# fig, ax = plt.subplots()
#
# plt.scatter(scaled_lhs[:, 0], scaled_lhs[:, 1])
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
# ax.set_title("LHS-Generated Styrene versus Polystyrene Volumes w/ Solvent")
# ax.set_xlabel("Volume Styrene (uL)")
# ax.set_ylabel("Volume Polystyrene (uL)")
#
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', pad=10)
#
# ax.grid(True, linestyle='-', linewidth=0.2, which='major', axis='both')
#
# plt.savefig(
#     r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Reports & Presentations\Introductory Report\Figures for Talk\3 factor LHS.png")
#
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(scaled_lhs[:, 0], scaled_lhs[:, 1], scaled_lhs[:, 2], marker='o')
#
# ax.set_xlabel('Styrene Volume (uL)')
# ax.set_ylabel('Polystyrene Volume (uL)')
# ax.set_zlabel('Solvent Volume (uL)')
# ax.set_title('LHS-Generated Styrene vs Polystyrene vs Solvent Volumes')
#
# plt.savefig(
#     r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Reports & Presentations\Introductory Report\Figures for Talk\3 factor LHS 3D.png")

# # Convert to a pandas DataFrame
# DF = pd.DataFrame(scaled_lhs, columns=['Styrene (µL)', 'Polystyrene (µL)', 'Solvent (µL)'])
#
# # Save to CSV
# DF.to_csv("Volumes.csv", index=False)
#
# solvent_volumes = scaled_lhs[:, 2]
# solvent_volumes = np.repeat(solvent_volumes, 2)
#
# styrene_volumes = scaled_lhs[:, 0]
# styrene_volumes = np.repeat(styrene_volumes, 2)
#
# polystyrene_volumes = scaled_lhs[:, 1]
# polystyrene_volumes = np.repeat(polystyrene_volumes, 2)
#
# processed_lhs = np.column_stack((styrene_volumes, polystyrene_volumes, solvent_volumes))
#
# # Convert to a pandas DataFrame
# DF = pd.DataFrame(processed_lhs, columns=['Styrene (µL)', 'Polystyrene (µL)', 'Solvent (µL)'])
#
# # Save to CSV
# DF.to_csv("Volumes_Duplicated.csv", index=False)


def log_msg(message):
    """Log a message with a timestamp."""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{current_time}] {message}")


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


def generate_lhs_design(num_samples, total_volume, step_size, num_factors):
    # Generate LHS design
    lhs = pyDOE2.lhs(num_factors, samples=num_samples)

    # Scale LHS design to the volume constraints
    scaled_lhs = lhs * (total_volume / num_factors)

    # Round to the nearest step_size and ensure minimum volume constraint
    scaled_lhs = np.round(scaled_lhs / step_size, 2) * step_size
    scaled_lhs = np.round(np.clip(scaled_lhs, step_size, None), 2)

    # Calculate the solvent volume as total_volume minus the sum of the components
    solvent_volumes = total_volume - np.sum(scaled_lhs, axis=1)

    # Combine the component volumes with the solvent volume as the last column
    scaled_lhs = np.column_stack((scaled_lhs, solvent_volumes))

    return scaled_lhs


def gen_volumes_csv(out_path, num_samples=46, total_volume=300, step_size=20, num_factors=2):
    # Generate and verify the LHS design
    while True:
        scaled_lhs = generate_lhs_design(num_samples, total_volume, step_size, num_factors)
        if np.all(np.sum(scaled_lhs, axis=1) == total_volume) and len(np.unique(scaled_lhs, axis=0)) == num_samples:
            print("VERIFIED: All samples are unique and sum to 300 uL.")
            break
        print("ERROR: Some samples are not unique or do not sum to 300 uL. Retrying...")

    # Define column names
    columns = [f'Component {i + 1}' for i in range(num_factors)] + ['Solvent']

    # Create DataFrame and save to CSV
    volumes = pd.DataFrame(scaled_lhs, columns=columns)
    current_time = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    volumes_file_path = out_path + f"\\Volumes_{current_time}.csv"
    volumes.to_csv(volumes_file_path, index=False)

    # Prepare duplicated volumes and additional solvent rows
    duplicated_volumes = np.repeat(scaled_lhs, 2, axis=0)
    duplicated_df = pd.DataFrame(duplicated_volumes, columns=columns)

    # Add four new rows with zeros in component columns and total_volume in the final column
    extra_rows = pd.DataFrame([[0] * num_factors + [total_volume]] * 4, columns=columns)
    duplicated_df = pd.concat([extra_rows, duplicated_df], ignore_index=True)

    # Save duplicated DataFrame to CSV
    duplicated_out_path = out_path + f"\\Duplicated_Volumes_{current_time}.csv"
    duplicated_df.to_csv(duplicated_out_path, index=False)

    print("\n" + duplicated_df.round(2).to_csv(index=False))

    return duplicated_df, duplicated_out_path


if __name__ == "__main__":
    df, csv_path = gen_volumes_csv(out_path=r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\Multivariable Experiments", num_factors=3)

    # Dictionary to hold volumes for each column
    volumes_dict = {}

    with open(csv_path, mode='r', newline="") as file:
        csv_reader = csv.DictReader(file)

        # Initialize lists in the dictionary for each column header
        for column in csv_reader.fieldnames:
            volumes_dict[column] = []

        # Iterate over rows and populate lists in the dictionary
        for row in csv_reader:
            for column, value in row.items():
                volumes_dict[column].append(float(value))

    # Example: Accessing lists by column name
    print(volumes_dict)

    for i in volumes_dict:
        print(volumes_dict[i])
