import numpy as np
import pyDOE2
import matplotlib.pyplot as plt
import pandas as pd

# Define constants
num_samples = 46  # the number of unique samples to be measured
total_volume = 300  # final volume in each well
step_size = 20  # minimum step size
num_factors = 2  # number of variables (styrene, polystyrene)


def generate_lhs_design(num_samples, total_volume, step_size, num_factors):
    # Generate LHS design
    lhs = pyDOE2.lhs(num_factors, samples=num_samples)

    # Scale LHS design to the volume constraints
    scaled_lhs = lhs * (total_volume / num_factors)

    # Round to the nearest step_size and ensure minimum volume constraint
    scaled_lhs = np.round(scaled_lhs / step_size, 2) * step_size
    scaled_lhs = np.clip(scaled_lhs, step_size, None)

    # Calculate the third column (solvent) as 300 minus the sum of the first two columns
    solvent_volumes = total_volume - np.sum(scaled_lhs, axis=1)

    # Combine the first two columns with the new solvent column
    scaled_lhs = np.column_stack((scaled_lhs, solvent_volumes))

    return scaled_lhs


# Generate the LHS design
scaled_lhs = generate_lhs_design(num_samples, total_volume, step_size, num_factors)

# Verification
# Check that all rows sum to total_volume
volume_sums = np.sum(scaled_lhs, axis=1)

while np.all(volume_sums != total_volume) or len(np.unique(scaled_lhs, axis=0)) != num_samples:
    print("ERROR: Some samples are not unique or do not sum to 300 uL")
    scaled_lhs = generate_lhs_design(num_samples, total_volume, step_size, num_factors)  # regenerate the design if doesn't pass checks
    volume_sums = np.sum(scaled_lhs, axis=1)

# Check that all rows sum to total_volume
volume_sums = np.sum(scaled_lhs, axis=1)
if np.all(volume_sums == total_volume):
    print("VERIFIED: All samples have a total volume of 300 units.")
else:
    print("ERROR: Some samples do not sum to 300 units.")

# Check that all rows are unique
if len(np.unique(scaled_lhs, axis=0)) == num_samples:
    print("VERIFIED: All samples are unique.")
else:
    print("ERROR: Some samples are not unique.")

# Print and plot the verified design
# print(scaled_lhs)
# print(volume_sums)
#
# Visualises the spread of sample points in 2D
# plt.scatter(scaled_lhs[:, 0], scaled_lhs[:, 1])
# plt.show()

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(scaled_lhs[:, 0], scaled_lhs[:, 1], scaled_lhs[:, 2], marker='o')
#
# ax.set_xlabel('Styrene Volume')
# ax.set_ylabel('Polystyrene Volume')
# ax.set_zlabel('Solvent Volume')
# ax.set_title('Experimental Runs - Styrene vs Polystyrene vs Acrylate Volume')
#
# plt.show()
#

# Convert to a pandas DataFrame
DF = pd.DataFrame(scaled_lhs, columns=['Styrene (µL)', 'Polystyrene (µL)', 'Solvent (µL)'])

# Save to CSV
DF.to_csv("Volumes.csv", index=False)

solvent_volumes = scaled_lhs[:, 2]
solvent_volumes = np.repeat(solvent_volumes, 2)

styrene_volumes = scaled_lhs[:, 0]
styrene_volumes = np.repeat(styrene_volumes, 2)

polystyrene_volumes = scaled_lhs[:, 1]
polystyrene_volumes = np.repeat(polystyrene_volumes, 2)

processed_lhs = np.column_stack((styrene_volumes, polystyrene_volumes, solvent_volumes))

# Convert to a pandas DataFrame
DF = pd.DataFrame(processed_lhs, columns=['Styrene (µL)', 'Polystyrene (µL)', 'Solvent (µL)'])

# Save to CSV
DF.to_csv("Volumes_Duplicated.csv", index=False)

