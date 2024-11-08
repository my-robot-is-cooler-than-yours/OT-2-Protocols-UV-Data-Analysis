import json
from datetime import datetime


def edit_json_data(file_path):
    # Define the offsets and spacing values
    x_offset = 15.8  # Offset to apply to wells in the last row (row G)
    y_offset = 15.7  # Offset to apply to wells in the first column (column 1)
    x_spacing = 19.8  # Spacing between adjacent columns (x direction)
    y_spacing = 19.33  # Spacing between adjacent rows (y direction)
    z_value = 48  # Z value (height of top of well) to apply to all wells
    diameter = 10  # Well diameter
    depth = 20  # Well depth

    # Load the JSON file into a dictionary
    with open(file_path, 'r') as file:
        well_data = json.load(file)

    # Extract json data
    metadata = well_data["metadata"]
    parameters = well_data["parameters"]
    wells = well_data["wells"]
    groups = well_data.get("groups", [])

    # Define the rows and columns in the plate based on the "ordering" array in your JSON
    rows = ['A', 'B', 'C', 'D']
    columns = range(1, 7)

    # Ensure metadata is saved correctly
    metadata["displayName"] = "LA Industries - Small Capped Tube Holder"
    metadata["displayVolumeUnits"] = "µL"
    parameters["loadName"] = "small_capped_tube_holder"

    # Apply x_offset to wells in the last row (row G)
    for col in columns:
        well = f'D{col}'
        if well in wells:
            wells[well]['x'] += x_offset

    # Apply y_offset to wells in the first column (column 1)
    for row in rows:
        well = f'{row}1'
        if well in wells:
            wells[well]['y'] += y_offset

    # Set parameters for all wells
    for row_idx, row in enumerate(rows):
        for col in columns:
            well = f'{row}{col}'
            if well in wells:
                wells[well]['x'] = x_offset + (col - 1) * x_spacing  # x starts at x_offset
                wells[well]['y'] = y_offset + (len(rows) - row_idx - 1) * y_spacing  # y starts at y_offset
                wells[well]['z'] = z_value
                wells[well]['diameter'] = diameter
                wells[well]['depth'] = depth

    # Ensure all wells are in the "groups" list and ordered by columns (A1, B1, C1, ..., A2, B2, etc.)
    if groups:
        all_wells = []
        for col in columns:
            for row in rows:
                well = f'{row}{col}'
                if well in wells:
                    all_wells.append(well)

        groups[0]["wells"] = all_wells  # Replace the wells list with the newly ordered list

    # Save the modified data back to the JSON file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(well_data, file, indent=4, ensure_ascii=False)

    print("Well data and groups updated and saved.")

if __name__ == "__main__":
    edit_json_data(r"C:\Users\Lachlan Alexander\AppData\Roaming\Opentrons\labware\small_capped_tube_holder.json")
