from opentrons import protocol_api

# Define metadata for protocol
metadata = {
    "apiLevel": "2.19",
    "protocolName": "LA Well Plate Test",
    "description": """Tests liquid transfer in 96 well plate.""",
    "author": "Lachlan Alexander"
    }

# Define the run function including all steps to be carried out by the robot
def run(protocol: protocol_api.ProtocolContext):
    
    # Load labware on deck
    tips_1 = protocol.load_labware("opentrons_96_tiprack_1000ul", 7)
    tips_2 = protocol.load_labware("opentrons_96_tiprack_300ul", 8)
    reservoir_1 = protocol.load_labware("nest_12_reservoir_15ml", 4)
    reservoir_2 = protocol.load_labware("nest_12_reservoir_15ml", 5)
    reservoir_3 = protocol.load_labware("nest_12_reservoir_15ml", 6)
    plate = protocol.load_labware("thermoscientificnunc_96_wellplate_400ul", 1)
    
    # Load pipettes
    right_pipette = protocol.load_instrument("p1000_single_gen2", "right", tip_racks=[tips_1])
    left_pipette = protocol.load_instrument("p300_single_gen2", "left", tip_racks=[tips_2])
    
    # Begin liquid handling steps
    
    # Distributes water to wells in row 1 of plate
    right_pipette.distribute(volume=100, source=reservoir_2["A1"], dest=plate.rows()[0], new_tip="once", trash = False, blow_out = True)