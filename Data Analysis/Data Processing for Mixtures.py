import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

in_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Evaporation Test\PS Evaporation Test Processed.csv"
out_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Evaporation Test"


def get_average(path_input):
    try:
        # Load data
        df = pd.read_csv(path_input)

        # Select only the numeric columns
        numeric_df = df.select_dtypes(include='number')

        # Group by every two rows and calculate the mean
        averaged_df = numeric_df.groupby(numeric_df.index // 4).mean()

        # Resetting the index if necessary
        averaged_df = averaged_df.reset_index(drop=True)

        # Merge the averaged numeric data back with the non-numeric data
        non_numeric_df = df.select_dtypes(exclude='number')
        non_numeric_df = non_numeric_df.groupby(non_numeric_df.index // 4).first().reset_index(drop=True)
        final_df = pd.concat([non_numeric_df, averaged_df], axis=1)

        print("Averaged data has been calculated.")

        return final_df

    except Exception as e:
        print(f"An error occurred: {e}")


def get_std_dev(path_input):
    try:
        # Load data
        df = pd.read_csv(path_input)

        # Select only the numeric columns
        numeric_df = df.select_dtypes(include='number')

        # Group by every two rows and calculate the mean
        std_dev_df = numeric_df.groupby(numeric_df.index // 4).std()

        # Resetting the index if necessary
        std_dev_df = std_dev_df.reset_index(drop=True)

        # Merge the std dev numeric data back with the non-numeric data
        non_numeric_df = df.select_dtypes(exclude='number')
        non_numeric_df = non_numeric_df.groupby(non_numeric_df.index // 4).first().reset_index(drop=True)
        final_df = pd.concat([non_numeric_df, std_dev_df], axis=1)

        print("Standard deviation data has been calculated.")

        return final_df

    except Exception as e:
        print(f"An error occurred: {e}")


# Get both the average and std dev dataframes
average_df = get_average(in_path)
std_df = get_std_dev(in_path)

average_df.to_csv(out_path + r'\PS Evaporation Test Averages.csv', index=False)
std_df.to_csv(out_path + r'\PS Evaporation Test Std dev.csv', index=False)

# Remove data that exceeds a certain %RSD
rsd_thresh = 100
indexes_to_remove = all[all['260'] > rsd_thresh].index

# Step 2: Remove rows from df1 that have these indexes and reassign to average_df variable
sorted_averages = average_df.drop(indexes_to_remove)

# sort averages by desired values
sorted_averages=sorted_averages.sort_values(by=['Wavelength'], ascending=[False])

# Number of data points to plot
num_samples = 4

# Prepare data
x = [int(i) for i in sorted_averages.columns[3:].values]
y = [sorted_averages.iloc[i, 3:].values for i in range(0,num_samples)]

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10,6))

# Plot each set of y-values
for i in range(len(y)):
    ax.plot(x, y[i], label=f'{sorted_averages.iloc[i,1]}')

# Customize axis limits
ax.set_xlim([220, 1000])
ax.set_ylim([-0.7, 1.5])

# Customize x-ticks
ax.set_xticks(np.arange(220, 1000, 10))

# Customize labels and title
ax.set_title("Absorbance Spectra for 0.25 mg/mL PS in BuOAc Over Time")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Absorbance (AU)")

# Optional: Add a legend if you want to label the lines
ax.legend()

# Display the plot
plt.show()
