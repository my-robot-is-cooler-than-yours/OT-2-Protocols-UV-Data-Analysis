from opentrons import protocol_api

# Define metadata for protocol
metadata = {
    "apiLevel": "2.19",
    "protocolName": "Evaporation Experiment",
    "description": """
    Dispenses rep of 300 uL along columns.
    Column 12 is blank/solvent.
    Pauses for even period of time over 40 min period depending on number of samples.
    Number of reps and samples is customisable in script.
    """,
    "author": "Lachlan Alexander",
    "date last modified": "04-Sep-2024"
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

num_reps=4 # Rows of duplicate samples
num_samples=4 # Columns of the variable samples
pause_time=40/4 # Time robot pauses between dispenses

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

    # Puncture holes in foil before dispensing liquid
    right_pipette.distribute(
        volume=100,
        source=reservoirs[0].wells()[1],
        dest=[plates[0].rows()[1][:num_samples],
              plates[0].rows()[2][:num_samples],
              plates[0].rows()[3][:num_samples],
              plates[0].rows()[4][:num_samples],
              plates[0].columns()[-1][:num_reps]
              ]
    )

    # Distribute 300 uL analyte to columns 1-4 with rows 1-4 being repetitions
    for i in range(num_reps):
        right_pipette.distribute(
            volume=300,
            source=reservoirs[0].wells()[5],
            dest=plates[0].rows()[i][:num_samples],
            blow_out=True,
            blowout_location="source well"
        )

        # Pause protocol for 10 mins for evaporation time
        protocol.delay(minutes=pause_time)

    # Distribute blank to column 12
    right_pipette.distribute(
        volume=300,
        source=reservoirs[0].wells()[0],
        dest=plates[0].columns()[-1][:num_reps],
        blow_out=True,
        blowout_location="source well"
    )