from opentrons import protocol_api

# Define metadata for protocol
metadata = {
    "apiLevel": "2.20",
    "protocolName": "OT-2 Test Protocol",
    "description": """Test protocol, picks up and puts down a tip with P300 pipette.""",
    "author": "Lachlan Alexander"
    }

# Constants
PIPETTE_R_NAME = 'p1000_single_gen2'
PIPETTE_L_NAME = 'p300_single_gen2'

R_PIP_TIPRACK_SLOTS = [3]
R_PIP_TIPRACK_LOADNAME = 'opentrons_96_tiprack_1000ul'

L_PIP_TIPRACK_SLOTS = []
L_PIP_TIPRACK_LOADNAME = 'opentrons_96_tiprack_300ul'

RESERVOIR_SLOTS = []
RESERVOIR_LOADNAME = 'nest_12_reservoir_15ml'

WELL_PLATE_SLOTS = []
# WELL_PLATE_LOADNAME = 'thermoscientificnunc_96_wellplate_400ul'


# Define the run function including all steps to be carried out by the robot
def run(protocol: protocol_api.ProtocolContext):
    
    # Load labware on deck
    r_tipracks = [protocol.load_labware(R_PIP_TIPRACK_LOADNAME, slot) for slot in R_PIP_TIPRACK_SLOTS]
    l_tipracks = [protocol.load_labware(L_PIP_TIPRACK_LOADNAME, slot) for slot in L_PIP_TIPRACK_SLOTS]

    # # Load pipettes
    right_pipette = protocol.load_instrument("p1000_single_gen2", "right", tip_racks=r_tipracks)
    left_pipette = protocol.load_instrument("p300_single_gen2", "left", tip_racks=l_tipracks)
    
    # Begin liquid handling steps
    
    # Pick up and put down
    right_pipette.pick_up_tip()
    right_pipette.drop_tip()

    protocol.comment("Protocol Finished")