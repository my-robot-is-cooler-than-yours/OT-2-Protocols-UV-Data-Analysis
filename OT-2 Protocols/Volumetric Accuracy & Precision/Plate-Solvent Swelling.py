from opentrons import protocol_api

# Define metadata for protocol
metadata = {
    "apiLevel": "2.19",
    "protocolName": "Plate-Solvent Swelling Experiment",
    "description": """
    Dispenses solvent to plate with timed gap between dispenses.
    Designed to track the effect of the solvent (DMSO, BuOAc) on the absorbance baseline of the plate with time.
    Recommend adapting protocol to deal with viscous liquids. Refer to opentrons documentation.
    """,
    "author": "Lachlan Alexander",
    "date last modified": "09-Sep-2024",
    "change log": "-"
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
# WELL_PLATE_LOADNAME: str = 'greiner_96_wellplate_300ul'
WELL_PLATE_LOADNAME: str = 'biorad_96_wellplate_200ul_pcr'

num_reps=4 # Rows of duplicate samples
num_samples=8 # Columns of the variable samples
pause_time=30 # Time robot pauses between dispenses

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

    # Distribute analyte to columns 1-4 with rows 1-4 being repetitions
    for i in range(num_samples):
        right_pipette.distribute(
            volume=300,
            source=reservoirs[0].wells()[5],
            dest=plates[0].columns()[i][:num_reps],
            blow_out=True,
            blowout_location="source well"
        )

        # Pause protocol for specified time
        protocol.delay(minutes=pause_time)

    # Distribute time zero sample to column 11
    right_pipette.distribute(
        volume=300,
        source=reservoirs[0].wells()[0],
        dest=plates[0].columns()[-2][:num_reps],
        blow_out=True,
        blowout_location="source well"
    )

    # Distribute blank to column 12
    right_pipette.distribute(
        volume=300,
        source=reservoirs[0].wells()[0],
        dest=plates[0].columns()[-1][:num_reps],
        blow_out=True,
        blowout_location="source well"
    )

