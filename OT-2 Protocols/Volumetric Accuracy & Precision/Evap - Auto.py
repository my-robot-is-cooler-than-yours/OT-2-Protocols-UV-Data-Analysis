from opentrons import protocol_api
import json

# Define metadata for protocol
metadata = {
    "apiLevel": "2.20",
    "protocolName": "Evap - Auto",
    "description": """
    Dispenses 300 uL of solvent into wells for evaporation testing. Works with auto setup.
    """,
    "author": "Lachlan Alexander",
    "date last modified": "30-Oct-2024",
    "change log": "Created"
}

# Constants
PIPETTE_R_NAME: str = 'p1000_single_gen2'
PIPETTE_L_NAME: str = 'p300_single_gen2'

R_PIP_TIPRACK_SLOTS: list = [8]
R_PIP_TIPRACK_LOADNAME: str = 'opentrons_96_tiprack_1000ul'

L_PIP_TIPRACK_SLOTS: list = [9]
L_PIP_TIPRACK_LOADNAME: str = 'opentrons_96_tiprack_300ul'

RESERVOIR_SLOTS: list = [2]
RESERVOIR_LOADNAME: str = 'nest_12_reservoir_15ml'

WELL_PLATE_SLOTS: list = [3]
WELL_PLATE_LOADNAME: str = 'greiner_96_wellplate_300ul'
# WELL_PLATE_LOADNAME: str = 'biorad_96_wellplate_200ul_pcr'  # simulation only, remove for actual protocol

well_height = 10.9

# Begin protocol


def run(protocol: protocol_api.ProtocolContext):
    # Load predefined labware on deck
    r_tipracks: list = [protocol.load_labware(R_PIP_TIPRACK_LOADNAME, slot) for slot in R_PIP_TIPRACK_SLOTS]
    l_tipracks: list = [protocol.load_labware(L_PIP_TIPRACK_LOADNAME, slot) for slot in L_PIP_TIPRACK_SLOTS]
    reservoirs: list = [protocol.load_labware(RESERVOIR_LOADNAME, slot) for slot in RESERVOIR_SLOTS]

    def load_custom_labware(file_path, location):
        with open(file_path) as labware_file:
            labware_def = json.load(labware_file)
        return protocol.load_labware_from_definition(labware_def, location)

    try:
        # Load custom labware on deck
        plates = [load_custom_labware("/data/user_storage/labware/slot 4 working.json", 4),
                  load_custom_labware("/data/user_storage/labware/slot 7 working ordered.json", 7),
                  ]

    except Exception as e:
        # Load definition stored on PC if fails
        plates = [protocol.load_labware(WELL_PLATE_LOADNAME, slot) for slot in WELL_PLATE_SLOTS]

    # Load labware offsets (FROM OPENTRONS APP - PLEASE ENSURE THESE ARE FILLED BEFORE RUNNING EXECUTE)
    r_tipracks[0].set_offset(x=0.10, y=-1.50, z=-1.00)
    l_tipracks[0].set_offset(x=0.00, y=0.70, z=0.00)
    reservoirs[0].set_offset(x=0.00, y=0.00, z=0.00)
    plates[0].set_offset(x=1.10, y=0.10, z=-8.50)  # Please for the love of god make sure these two are set properly
    plates[1].set_offset(x=1.60, y=0.40, z=-8.70)

    # Load pipettes
    right_pipette = protocol.load_instrument(PIPETTE_R_NAME, "right", tip_racks=r_tipracks)
    left_pipette = protocol.load_instrument(PIPETTE_L_NAME, "left", tip_racks=l_tipracks)

    # Begin liquid handling steps

    # Prepare well positions
    plate_wells = []  # Reset list
    for row1, row2 in zip(plates[0].columns(), plates[1].columns()):
        plate_wells.extend(row1)  # Add wells from the current row of slot 4 definition
        plate_wells.extend(row2)  # Add wells from the current row of slot 7 definition

    protocol.comment("Adding solvent to first well")
    right_pipette.distribute(
        volume=300,
        source=reservoirs[0].wells()[0],
        dest=plate_wells[0],
        blow_out=True,
        blowout_location="source well"
    )

    protocol.comment("Adding styrene to wells")
    right_pipette.distribute(
        volume=300,
        source=reservoirs[0].wells()[-1],
        dest=[well.top() for well in plate_wells[1:25]],
        blow_out=True,
        blowout_location="source well"
    )

    protocol.comment("Protocol Finished")
