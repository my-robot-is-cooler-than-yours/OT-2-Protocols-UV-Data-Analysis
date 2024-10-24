import json


def edit_json_data(file_path):
    # Define the offsets and spacing values
    x_offset = 25  # Offset to apply to wells in the last row (row G)
    y_offset = 5.0  # Offset to apply to wells in the first column (column 1)
    spacing = 9.0  # Spacing between wells in both x and y directions
    z_value = 79  # Z value (height of top of well) to apply to all wells
    diameter = 6.96  # Well diameter
    depth = 10.90  # Well depth


    # Load the JSON file into a dictionary
    with open(file_path, 'r') as file:
        well_data = json.load(file)

    # Extract json data
    metadata = well_data["metadata"]
    parameters = well_data["parameters"]
    wells = well_data["wells"]
    groups = well_data.get("groups", [])

    # Define the rows and columns in the plate based on the "ordering" array in your JSON
    rows = ['A', 'B', 'C', 'D', 'E']
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    columns = range(1, 9)  # Assuming 12 columns

    # Ensure metadata is saved correctly
    metadata["displayName"] = "LA Industries - Plate Reader Slot 7"
    metadata["displayVolumeUnits"] = "µL"
    parameters["loadName"] = "plate_reader_slot_7"

    # Apply x_offset to wells in the last row (row G)
    for col in columns:
        well = f'G{col}'
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
                wells[well]['x'] = x_offset + (col - 1) * spacing  # x starts at x_offset
                wells[well]['y'] = y_offset + (len(rows) - row_idx - 1) * spacing  # y starts at y_offset
                wells[well]['z'] = z_value
                wells[well]['diameter'] = diameter
                wells[well]['depth'] = depth

    # Ensure all wells are in the "groups" list and order by columns (A1, B1, C1, ..., A2, B2, etc.)
    # If the "groups" list is not empty, we'll assume the first entry is where we need to add the wells
    if groups:
        all_wells = []
        for col in columns:
            for row in rows:
                well = f'{row}{col}'
                if well in wells:
                    all_wells.append(well)

        groups[0]["wells"] = all_wells  # Replace the wells list with the newly ordered list

    # Save the modified data back to the JSON file
    with open('well_data_modified.json', 'w', encoding='utf-8') as file:
        json.dump(well_data, file, indent=4, ensure_ascii=False)

    print("Well data and groups updated and saved to well_data_modified.json.")


if __name__ == "__main__":
    edit_json_data(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Honours Python Main\Misc Scripts\laindustries_slot_7_40_wellplate_400ul - Copy.json")
