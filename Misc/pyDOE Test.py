import pyDOE2 as pyDOE
import numpy as np
import scipy
import matplotlib.pyplot as plt

# # Define the levels of styrene and polystyrene concentrations
# styrene_levels = [0.1, 0.5, 1.0]  # Example concentrations in some unit
# polystyrene_levels = [0.1, 0.5, 1.0]  # Example concentrations in some unit
#
# # Create a full factorial design
# design = pyDOE.fullfact([len(styrene_levels), len(polystyrene_levels)])
#
# # Map the design to the actual concentration levels
# styrene_concentrations = np.array(styrene_levels)[design[:, 0].astype(int)]
# polystyrene_concentrations = np.array(polystyrene_levels)[design[:, 1].astype(int)]
#
# # Combine the concentrations into a single matrix
# experimental_runs = np.column_stack((styrene_concentrations, polystyrene_concentrations))
#
# # Print the experimental runs
# print("Experimental Runs (Styrene, Polystyrene):")
# print(experimental_runs)
#
# # Placeholder for your UV-Vis absorbance data
# # You will need to replace this with your actual data collection
# absorbance_data = np.random.random(len(experimental_runs))
#
# # Combine experimental runs with absorbance data
# results = np.column_stack((experimental_runs, absorbance_data))
#
# # Print the results
# print("Results (Styrene, Polystyrene, Absorbance):")
# print(results)

# Latin hypercube model
# Define the number of combinations to investigate
num_combinations = 46

# Define the range of styrene and polystyrene concentrations
styrene_min, styrene_max = 0.1, 1.0  # Example range
polystyrene_min, polystyrene_max = 0.1, 1.0  # Example range

# Generate a Latin Hypercube Sample
lhs = pyDOE.lhs(2, samples=num_combinations)

# Scale the samples to the defined concentration ranges
styrene_concentrations = lhs[:, 0] * (styrene_max - styrene_min) + styrene_min
polystyrene_concentrations = lhs[:, 1] * (polystyrene_max - polystyrene_min) + polystyrene_min

# Combine the concentrations into a single matrix
experimental_runs = np.column_stack((styrene_concentrations, polystyrene_concentrations))

print(np.shape(experimental_runs))

# Plot the experimental runs
plt.figure(figsize=(8, 6))
plt.scatter(styrene_concentrations, polystyrene_concentrations, marker='o', color='b', label='Experimental Runs')
plt.xlabel('Styrene Concentration')
plt.ylabel('Polystyrene Concentration')
plt.title('Experimental Runs - Styrene vs Polystyrene Concentrations')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




