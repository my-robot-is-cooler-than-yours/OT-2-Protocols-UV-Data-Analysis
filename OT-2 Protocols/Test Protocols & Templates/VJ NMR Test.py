from opentrons import protocol_api

# Define constants
total_volume = 350  # final volume in each well
step_size = 20  # minimum step size
well_height = 10.9  # mm from top to bottom of well

# Define metadata for protocol
metadata = {
    "apiLevel": "2.19",
    "protocolName": "VJ NMR Test",
    "description": """
    """,
    "author": "Lachlan Alexander",
    "date last modified": "10-Oct-2024",
    "change log": ""
}

# Constants
PIPETTE_R_NAME: str = 'p1000_single_gen2'
PIPETTE_L_NAME: str = 'p300_single_gen2'

R_PIP_TIPRACK_SLOTS: list = [10]
R_PIP_TIPRACK_LOADNAME: str = 'opentrons_96_tiprack_1000ul'

L_PIP_TIPRACK_SLOTS: list = [1]
L_PIP_TIPRACK_LOADNAME: str = 'opentrons_96_tiprack_300ul'

RESERVOIR_SLOTS: list = [2]
RESERVOIR_LOADNAME: str = 'nest_12_reservoir_15ml'

WELL_PLATE_SLOTS: list = [3]
WELL_PLATE_LOADNAME: str = 'vjptyltdnmrholder_48_wellplate_1000ul'
# WELL_PLATE_LOADNAME: str = 'biorad_96_wellplate_200ul_pcr'  # simulation only, remove for actual protocol


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

    # Step 1: Dispense 350 uL of solvent into the first well (blank)
    protocol.comment("!!!Dispensing 350 uL of solvent into the first well!!!")
    left_pipette.transfer(
        350,  # Volume
        reservoirs[0].wells()[0],  # Source (solvent well)
        plates[0].wells()[0],  # Destination (first well)
        new_tip='Always',
        blow_out=True,  # Blow out after dispensing
        blowout_location='destination well'  # Blow out in the destination well
    )