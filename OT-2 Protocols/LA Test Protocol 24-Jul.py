from opentrons import protocol_api

# Define metadata for protocol
metadata = {
    "apiLevel": "2.19",
    "protocolName": "LA OT-2 Protocol Template",
    "description": """Protocol template for the OT-2. Includes many useful liquid handling steps.""",
    "author": "Lachlan Alexander"
    }

# Constants
PIPETTE_R_NAME = 'p1000_single_gen2'
PIPETTE_L_NAME = 'p300_single_gen2'

R_PIP_TIPRACK_SLOTS = [7]
R_PIP_TIPRACK_LOADNAME = 'opentrons_96_tiprack_1000ul'

L_PIP_TIPRACK_SLOTS = [8]
L_PIP_TIPRACK_LOADNAME = 'opentrons_96_tiprack_300ul'

RESERVOIR_SLOTS = [4,5,6]
RESERVOIR_LOADNAME = 'nest_12_reservoir_15ml'

WELL_PLATE_SLOTS = [1]
WELL_PLATE_LOADNAME = 'thermoscientificnunc_96_wellplate_400ul'

# Define the run function including all steps to be carried out by the robot
def run(protocol: protocol_api.ProtocolContext):
    
    # Load labware on deck
    r_tipracks = [protocol.load_labware(R_PIP_TIPRACK_LOADNAME, slot) for slot in R_PIP_TIPRACK_SLOTS]
    l_tipracks = [protocol.load_labware(L_PIP_TIPRACK_LOADNAME, slot) for slot in L_PIP_TIPRACK_SLOTS]
        
    reservoir_1 = protocol.load_labware("nest_12_reservoir_15ml", 4)
    reservoir_2 = protocol.load_labware("nest_12_reservoir_15ml", 5)
    reservoir_3 = protocol.load_labware("nest_12_reservoir_15ml", 6)
    plate = protocol.load_labware("thermoscientificnunc_96_wellplate_400ul", 1)
    
    # Load pipettes
    right_pipette = protocol.load_instrument("p1000_single_gen2", "right", tip_racks=r_tipracks)
    left_pipette = protocol.load_instrument("p300_single_gen2", "left", tip_racks=l_tipracks)
    
    # Begin liquid handling steps
    
    # Transfer 100 uL from reservoir well A1 to well A3
    # Uses second pipette to transfer 200 uL between wells A1 and A3
    right_pipette.transfer(100,
                           reservoir_1["A1"],
                           reservoir_1["A3"]
                           )
    
    left_pipette.transfer(100,
                          reservoir_2["A1"],
                          reservoir_2["A3"]
                          )
    
    # Transfer 100 uL from reservoir well A1 to every remaining well 
    right_pipette.transfer(volume=100,
                           source=reservoir_1["A1"],
                           dest=reservoir_1.wells()[1:12],
                           new_tip="once",
                           trash=False,
                           blow_out=True,
                           blowout_location="source well"
                           )
    
    # This is more effectively done with the distribute function
    right_pipette.distribute(volume=100,
                             source=reservoir_2["A1"],
                             dest=reservoir_2.wells()[1:12],
                             new_tip="once",
                             trash=False,
                             blow_out=True
                             )
    # Note that when 'trash' is set to False and 'blow_out' is set to True, the robot will still blow excess liquid into trash during distribution

    # Aspirates from two separate reservoir wells with an air bubble between them, then dispenses them into separate wells (causes cross-contamination, demo only)
    right_pipette.pick_up_tip()
    
    right_pipette.aspirate(100, reservoir_1["A1"])
    right_pipette.air_gap(volume=100)
    right_pipette.aspirate(100, reservoir_2["A2"])
    right_pipette.dispense(150, reservoir_3["A1"])
    right_pipette.dispense(150, reservoir_3["A2"])
    right_pipette.blow_out()
    
    right_pipette.return_tip()
    
    # Transfer 100 uL from reservoir 1 to every second well on the well plate, starting with well A1
    # Note that just this took 28 mins in total
    for i in range(len(plate.wells())):
        if i%2==0:
            left_pipette.transfer(volume=100,
                                  source=reservoir_1["A1"],
                                  dest=plate.wells()[i],
                                  new_tip="once",
                                  trash=False,
                                  blow_out=True,
                                  blowout_location="source well",
                                  mix_after=(3,50),
                                  touch_tip=True)
            
            
         
            
            
            
            
            
            
            
            
            
            
            