from opentrons import protocol_api
import json

# Define metadata for protocol
metadata = {
    "apiLevel": "2.20",
    "protocolName": "OT-2 Test Protocol",
    "description": """Test protocol""",
    "author": "Lachlan Alexander",
    "date last modified": "21-Oct-2024",
    "change log": "Created"
}

# Define pipette names
PIPETTE_R_NAME = 'p1000_single_gen2'
PIPETTE_L_NAME = 'p300_single_gen2'

# Define labware names and locations
R_PIP_TIPRACK_SLOTS = [3]
R_PIP_TIPRACK_LOADNAME = 'opentrons_96_tiprack_1000ul'

L_PIP_TIPRACK_SLOTS = []
L_PIP_TIPRACK_LOADNAME = 'opentrons_96_tiprack_300ul'

RESERVOIR_SLOTS = []
RESERVOIR_LOADNAME = 'nest_12_reservoir_15ml'

WELL_PLATE_SLOTS = [6]
WELL_PLATE_LOADNAME = 'greiner_96_wellplate_300ul'


# Define the run function including all steps to be carried out by the robot
def run(protocol: protocol_api.ProtocolContext):
    # Load labware on deck
    r_tipracks = [protocol.load_labware(R_PIP_TIPRACK_LOADNAME, slot) for slot in R_PIP_TIPRACK_SLOTS]
    l_tipracks = [protocol.load_labware(L_PIP_TIPRACK_LOADNAME, slot) for slot in L_PIP_TIPRACK_SLOTS]

    # well_plates = [protocol.load_labware(WELL_PLATE_LOADNAME, slot) for slot in WELL_PLATE_SLOTS]

    def load_custom_labware(file_path, location):
        with open(file_path) as labware_file:
            labware_def = json.load(labware_file)
        return protocol.load_labware_from_definition(labware_def, location)

    well_plates = [load_custom_labware("/data/user_storage/labware/greiner_96_wellplate_300ul.json", 6)]

    # Load labware offsets (FROM OPENTRONS APP)
    r_tipracks[0].set_offset(x=0.30, y=-1.30, z=-0.90)
    well_plates[0].set_offset(x=-0.20, y=-1.50, z=0.00)

    # # Load pipettes
    right_pipette = protocol.load_instrument("p1000_single_gen2", "right", tip_racks=r_tipracks)
    left_pipette = protocol.load_instrument("p300_single_gen2", "left", tip_racks=l_tipracks)

    # Begin liquid handling steps

    # Pick up and put down
    right_pipette.pick_up_tip()
    right_pipette.move_to(well_plates[0].wells()[0].top())
    right_pipette.return_tip()

    protocol.comment("Protocol Finished")
