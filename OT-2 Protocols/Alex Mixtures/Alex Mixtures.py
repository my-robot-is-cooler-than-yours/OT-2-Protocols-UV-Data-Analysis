from opentrons import protocol_api
import csv

# Input raw CSV data for volumes of each liquid to be used. Should be a 92 x 3 matrix containing volume values in uL.

# Alex reagent volumes
csv_raw = '''
Poly,A,B,Base
200,20,60,20
200,20,60,20
200,20,60,20
200,20,60,20
200,20,60,20
200,25,55,20
200,25,55,20
200,25,55,20
200,25,55,20
200,25,55,20
200,30,50,20
200,30,50,20
200,30,50,20
200,30,50,20
200,30,50,20
200,35,45,20
200,35,45,20
200,35,45,20
200,35,45,20
200,35,45,20
200,40,40,20
200,40,40,20
200,40,40,20
200,40,40,20
200,40,40,20
200,45,35,20
200,45,35,20
200,45,35,20
200,45,35,20
200,45,35,20
200,50,30,20
200,50,30,20
200,50,30,20
200,50,30,20
200,50,30,20
200,55,25,20
200,55,25,20
200,55,25,20
200,55,25,20
200,55,25,20
200,60,20,20
200,60,20,20
200,60,20,20
200,60,20,20
200,60,20,20
'''

# Parse the CSV data
csv_data = csv_raw.strip().splitlines()
csv_reader = csv.DictReader(csv_data)

poly_volumes = []
a_volumes = []
b_volumes = []
base_volumes = []

for row in csv_reader:
    poly_volumes.append(float(row['Poly']))
    a_volumes.append(float(row['A']))
    b_volumes.append(float(row['B']))
    base_volumes.append(float(row['Base']))

# Define constants
num_samples = len(poly_volumes)  # the number of unique samples to be measured
total_volume = 300  # final volume in each well
step_size = 20  # minimum step size
num_factors = 2  # number of variables (styrene, polystyrene)
well_height = 10.9  # mm from top to bottom of well

# Define metadata for protocol
metadata = {
    "apiLevel": "2.19",
    "protocolName": "Alex Mixtures for Funky Polymer",
    "description": """
    From CSV input, pipettes samples across columns with repetitions in each row.
    """,
    "author": "Lachlan Alexander",
    "date last modified": "24-Sep-2024",
    "change log": "Corrected for cross contamination and mixing. Corrected for well dispense height to save tips."
}

# Constants
PIPETTE_R_NAME: str = 'p1000_single_gen2'
PIPETTE_L_NAME: str = 'p300_single_gen2'

R_PIP_TIPRACK_SLOTS: list = [10]
R_PIP_TIPRACK_LOADNAME: str = 'opentrons_96_tiprack_1000ul'

L_PIP_TIPRACK_SLOTS: list = [11]
L_PIP_TIPRACK_LOADNAME: str = 'opentrons_96_tiprack_300ul'

RESERVOIR_SLOTS: list = [8]
RESERVOIR_LOADNAME: str = 'nest_12_reservoir_15ml'

WELL_PLATE_SLOTS: list = [9]
WELL_PLATE_LOADNAME: str = 'greiner_96_wellplate_300ul'
# WELL_PLATE_LOADNAME: str = 'biorad_96_wellplate_200ul_pcr' # simulation only, remove for actual protocol


# Begin protocol

def run(protocol: protocol_api.ProtocolContext):
    # Load labware on deck
    r_tipracks: list = [protocol.load_labware(R_PIP_TIPRACK_LOADNAME, slot) for slot in R_PIP_TIPRACK_SLOTS]
    l_tipracks: list = [protocol.load_labware(L_PIP_TIPRACK_LOADNAME, slot) for slot in L_PIP_TIPRACK_SLOTS]
    reservoirs: list = [protocol.load_labware(RESERVOIR_LOADNAME, slot) for slot in RESERVOIR_SLOTS]
    plates: list = [protocol.load_labware(WELL_PLATE_LOADNAME, slot) for slot in WELL_PLATE_SLOTS]

    # Load pipettes
    right_pipette = protocol.load_instrument(PIPETTE_R_NAME, "right", tip_racks=r_tipracks)
    left_pipette = protocol.load_instrument(PIPETTE_L_NAME, "left", tip_racks=l_tipracks)

    # Begin liquid handling steps

    # Prepare well positions
    sample_wells = [well for col in plates[0].columns()[:9] for well in col[:5]]  # Columns 1-9, rows 1-5

    # Ensure target wells cover the required number of samples (e.g., num_samples = 45)
    target_wells = sample_wells[:num_samples]

    protocol.comment("Dispensing Polymer!")
    # Step 1: Distribute polymer to every well
    right_pipette.distribute(
        volume=poly_volumes,
        source=reservoirs[0].wells()[1],
        dest=target_wells,
        blow_out=True,
        blowout_location="source well"
    )

    # Step 2: Distribute A
    protocol.comment("Dispensing Reagent A!")
    right_pipette.distribute(
        volume=a_volumes,
        source=reservoirs[0].wells()[2],
        dest=[well.top() for well in target_wells],
        blow_out=True,
        blowout_location="source well"
    )

    # Step 3: Distribute B
    protocol.comment("Dispensing Reagent B!")
    right_pipette.distribute(
        volume=b_volumes,
        source=reservoirs[0].wells()[3],
        dest=[well.top() for well in target_wells],
        blow_out=True,
        blowout_location="source well"
    )

    # Step 4: Transfer Base + mix
    protocol.comment("Dispensing Base and Mixing 100 ul x 3 reps!")
    left_pipette.transfer(
        volume=base_volumes,
        source=reservoirs[0].wells()[4],
        dest=[well.bottom(well_height//2) for well in target_wells],
        new_tip="always",
        mix_after=(3, 100),
        blow_out=True,
        blowout_location="destination well"
    )
