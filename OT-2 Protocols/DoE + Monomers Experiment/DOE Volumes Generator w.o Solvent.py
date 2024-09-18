import numpy as np
import pyDOE2
import matplotlib.pyplot as plt
import pandas as pd

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

    # Visualise the spread of sample points in 2D (for 2 factors only)
    if num_factors == 2:
        plt.scatter(scaled_lhs[:, 0], scaled_lhs[:, 1])
        plt.xlabel('Factor 1')
        plt.ylabel('Factor 2')
        plt.show()

    # Convert to a pandas DataFrame
    DF = pd.DataFrame(scaled_lhs, columns=['Styrene (µL)', 'Polystyrene (µL)'])

    # Save to CSV
    DF.to_csv(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Honours Python Main\OT-2 Protocols\DoE + Monomers Experiment\Volumes 18-Sep.csv", index=False)

    styrene_volumes = scaled_lhs[:, 0]
    styrene_volumes = np.repeat(styrene_volumes, 2)

    polystyrene_volumes = scaled_lhs[:, 1]
    polystyrene_volumes = np.repeat(polystyrene_volumes, 2)

    processed_lhs = np.column_stack((styrene_volumes, polystyrene_volumes))

    # Convert to a pandas DataFrame
    DF = pd.DataFrame(processed_lhs, columns=['Styrene (µL)', 'Polystyrene (µL)'])

    # Save to CSV
    DF.to_csv(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Honours Python Main\OT-2 Protocols\DoE + Monomers Experiment\Volumes 18-Sep Duplicated.csv", index=False)

if __name__ == "__main__":
    main()