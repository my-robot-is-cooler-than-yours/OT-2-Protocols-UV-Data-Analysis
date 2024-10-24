from opentrons import protocol_api

# Define metadata for protocol
metadata = {
    "apiLevel": "2.20",
    "protocolName": "Labware Offset Protocol",
    "description": """For use in determining labware offsets for labware in different locations on the deck.""",
    "author": "Lachlan Alexander",
    "date last modified": "21-Oct-2024",
    "change log": "Created"
}

# Define pipette names
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


# Define the run function including all steps to be carried out by the robot
def run(protocol: protocol_api.ProtocolContext):
    """
    Used to determine the labware offsets for both predetermined and custom labware used on the OT-2 deck.
    Ensure that all custom labware has been defined in line with Opentrons guidelines and
    labware name/position details are correctly entered to the above variables.

    :param protocol:
    :return: None
    """
    # Load labware on deck
    r_tipracks = [protocol.load_labware(R_PIP_TIPRACK_LOADNAME, slot) for slot in R_PIP_TIPRACK_SLOTS]
    l_tipracks = [protocol.load_labware(L_PIP_TIPRACK_LOADNAME, slot) for slot in L_PIP_TIPRACK_SLOTS]
    reservoirs = [protocol.load_labware(RESERVOIR_LOADNAME, slot) for slot in RESERVOIR_SLOTS]
    # well_plates = [protocol.load_labware(WELL_PLATE_LOADNAME, slot) for slot in WELL_PLATE_SLOTS]
    well_plates = [protocol.load_labware("laindustries_slot_4_40_wellplate_400ul", 4),
                   protocol.load_labware("laindustries_slot_7_40_wellplate_400ul", 7)
                   ]

    # Load pipettes
    right_pipette = protocol.load_instrument("p1000_single_gen2", "right", tip_racks=r_tipracks)
    left_pipette = protocol.load_instrument("p300_single_gen2", "left", tip_racks=l_tipracks)

    # Pick up, move, and put down at various labware to get offsets
    left_pipette.pick_up_tip()
    left_pipette.move_to(well_plates[0].wells()[0].top())
    left_pipette.move_to(well_plates[1].wells()[0].top())
    left_pipette.move_to(reservoirs[0].wells()[0].top())
    left_pipette.return_tip()

    right_pipette.pick_up_tip()
    right_pipette.return_tip()

    protocol.comment("Protocol Finished")
