import pyDOE2 as pyDOE
import numpy as np
import matplotlib.pyplot as plt

# Define the number of combinations to investigate
num_combinations = 46

# Define the range of styrene and polystyrene concentrations
# Since they sum up to 1, we only need to define one range, the other will be complementary.
concentration_min, concentration_max = 0.0, 1.0

# Generate a Latin Hypercube Sample
lhs = pyDOE.lhs(1, samples=num_combinations)

# Scale the samples to the defined concentration ranges
styrene_concentrations = lhs[:, 0] * (concentration_max - concentration_min) + concentration_min

# Calculate polystyrene concentrations to ensure they sum up to 1
polystyrene_concentrations = 1 - styrene_concentrations

# Combine the concentrations into a single matrix
experimental_runs = np.column_stack((styrene_concentrations, polystyrene_concentrations))

# Print the experimental runs
print("Experimental Runs (Styrene, Polystyrene):")
print(experimental_runs)

# Placeholder for your UV-Vis absorbance data
# You will need to replace this with your actual data collection
absorbance_data = np.random.random(len(experimental_runs))

# Combine experimental runs with absorbance data
results = np.column_stack((experimental_runs, absorbance_data))

# Print the results
print("Results (Styrene, Polystyrene, Absorbance):")
print(results)

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
