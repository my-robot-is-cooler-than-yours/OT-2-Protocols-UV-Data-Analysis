from opentrons import protocol_api

# Define metadata for protocol
metadata = {
    "apiLevel": "2.19",
    "protocolName": "LA OT-2 Protocol Template",
    "description": """Protocol template for the OT-2. Includes many useful liquid handling steps.""",
    "author": "Lachlan Alexander"
}

# Constants
PIPETTE_R_NAME: str = 'p1000_single_gen2'
PIPETTE_L_NAME: str = 'p300_single_gen2'

R_PIP_TIPRACK_SLOTS: list = [7]
R_PIP_TIPRACK_LOADNAME: str = 'opentrons_96_tiprack_1000ul'

L_PIP_TIPRACK_SLOTS: list = [8]
L_PIP_TIPRACK_LOADNAME: str = 'opentrons_96_tiprack_300ul'

RESERVOIR_SLOTS: list = [4, 5, 6]
RESERVOIR_LOADNAME: str = 'nest_12_reservoir_15ml'

WELL_PLATE_SLOTS: list = [1]
WELL_PLATE_LOADNAME: str = 'thermoscientificnunc_96_wellplate_2000ul'

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
    
    # Transfer 100 uL from reservoir well A1 to well A3 in the same reservoir
    right_pipette.transfer(100, reservoirs[0]["A1"], reservoirs[0]["A3"])
    left_pipette.transfer(100, reservoirs[1]["A1"], reservoirs[1]["A3"])
    
    # Transfer 100 uL from reservoir well A1 to every remaining well in the same reservoir
    right_pipette.transfer(
        volume=100,
        source=reservoirs[0]["A1"],
        dest=reservoirs[0].wells()[1:12],
        new_tip="once",
        blow_out=True,
        blowout_location="source well"
    )
    
    # This is more effectively done with the distribute function
    right_pipette.distribute(
        volume=100,
        source=reservoirs[1]["A1"],
        dest=reservoirs[1].wells()[1:12],
        new_tip="once",
        blow_out=True,
        blowout_location="source well"
    )

    # Aspirates from two separate reservoir wells with an air bubble between them, then dispenses them into separate wells (demonstration only)
    right_pipette.pick_up_tip()
    
    right_pipette.aspirate(100, reservoirs[0]["A1"])
    right_pipette.air_gap(volume=100)
    right_pipette.aspirate(100, reservoirs[1]["A2"])
    right_pipette.dispense(150, reservoirs[2]["A1"])
    right_pipette.dispense(150, reservoirs[2]["A2"])
    right_pipette.blow_out()
    
    right_pipette.drop_tip()  # Ensure proper tip handling
    
    # Transfer 100 uL from reservoir 1 to every second well on the well plate, starting with well A1
    for i in range(0, len(plates[0].wells()), 2):
        left_pipette.transfer(
            volume=100,
            source=reservoirs[0]["A1"],
            dest=plates[0].wells()[i],
            new_tip="always",  # Ensuring a new tip for each transfer to avoid contamination
            blow_out=True,
            blowout_location="source well",
            mix_after=(3, 50),
            touch_tip=True
        )


                             
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
