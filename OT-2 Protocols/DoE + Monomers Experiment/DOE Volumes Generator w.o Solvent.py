import numpy as np
import pyDOE2
import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt
import pandas as pd

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
num_factors = 2  # number of variables (e.g., styrene, polystyrene, solvent)


def generate_lhs_design(num_samples, total_volume, step_size, num_factors):
    # Generate LHS design
    lhs = pyDOE2.lhs(num_factors, samples=num_samples)

    # Scale LHS design to the volume constraints
    scaled_lhs = lhs * total_volume

    # Round to the nearest step_size and ensure minimum volume constraint
    scaled_lhs = np.round(scaled_lhs / step_size, 2) * step_size
    scaled_lhs = np.clip(scaled_lhs, step_size, None)

    # Adjust volumes to ensure they sum up to total_volume
    row_sums = np.sum(scaled_lhs, axis=1)
    adjustment_factors = total_volume / row_sums
    scaled_lhs = (scaled_lhs.T * adjustment_factors).T

    # Round again after adjustment to match step_size
    scaled_lhs = np.round(scaled_lhs / step_size, 2) * step_size
    scaled_lhs = np.clip(scaled_lhs, step_size, None)

    return scaled_lhs


def main():
    # Generate the LHS design
    scaled_lhs = generate_lhs_design(num_samples, total_volume, step_size, num_factors)

    # Verification
    volume_sums = np.sum(scaled_lhs, axis=1)

    while np.any(volume_sums != total_volume) or len(np.unique(scaled_lhs, axis=0)) != num_samples:
        print("ERROR: Some samples are not unique or do not sum to total volume.")
        scaled_lhs = generate_lhs_design(num_samples, total_volume, step_size, num_factors)
        volume_sums = np.sum(scaled_lhs, axis=1)

    # Check if all rows sum to total_volume
    if np.all(volume_sums == total_volume):
        print("VERIFIED: All samples have a total volume of 300 units.")
    else:
        print("ERROR: Some samples do not sum to 300 units.")

    # Check if all rows are unique
    if len(np.unique(scaled_lhs, axis=0)) == num_samples:
        print("VERIFIED: All samples are unique.")
    else:
        print("ERROR: Some samples are not unique.")

    # Print and plot the verified design
    print(scaled_lhs)
    print(volume_sums)

    fig, ax = plt.subplots()
    # Visualise the spread of sample points in 2D (for 2 factors only)
    if num_factors == 2:
        plt.scatter(scaled_lhs[:, 0], scaled_lhs[:, 1])

        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_visible(False)

        # Change spine width
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.5)

        ax.set_title("LHS-Generated Styrene versus Polystyrene Volumes")
        ax.set_xlabel("Volume Styrene (uL)")
        ax.set_ylabel("Volume Polystyrene (uL)")

        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', direction='in', pad=10)

        ax.grid(True, linestyle='-', linewidth=0.2, which='major', axis='both')

        plt.savefig(
            r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Reports & Presentations\Introductory Report\Figures for Talk\2 factor LHS.png")

    # # Convert to a pandas DataFrame
    # DF = pd.DataFrame(scaled_lhs, columns=['Styrene (µL)', 'Polystyrene (µL)'])
    #
    # # Save to CSV
    # DF.to_csv(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Honours Python Main\OT-2 Protocols\DoE + Monomers Experiment\Volumes No Solvent 23-Sep.csv", index=True)
    #
    # styrene_volumes = scaled_lhs[:, 0]
    # styrene_volumes = np.repeat(styrene_volumes, 2)
    #
    # polystyrene_volumes = scaled_lhs[:, 1]
    # polystyrene_volumes = np.repeat(polystyrene_volumes, 2)
    #
    # processed_lhs = np.column_stack((styrene_volumes, polystyrene_volumes))
    #
    # # Convert to a pandas DataFrame
    # DF = pd.DataFrame(processed_lhs, columns=['Styrene (µL)', 'Polystyrene (µL)'])
    #
    # # Save to CSV
    # DF.to_csv(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Honours Python Main\OT-2 Protocols\DoE + Monomers Experiment\Volumes No Solvent 23-Sep Duplicated.csv", index=True)


if __name__ == "__main__":
    main()
