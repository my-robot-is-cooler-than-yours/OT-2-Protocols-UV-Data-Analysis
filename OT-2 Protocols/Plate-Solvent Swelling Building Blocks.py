from opentrons import protocol_api

# Define metadata for protocol
metadata = {
    "apiLevel": "2.19",
    "protocolName": "Plate-Solvent Swelling Experiment",
    "description": """
    Dispenses solvent to plate with timed gap between dispenses.
    Designed to track the effect of the solvent (DMSO, BuOAc) on the absorbance baseline of the plate with time.
    Recommend adapting protocol to deal with viscous liquids. Refer to opentrons documentation.
    Uses building block commands.
    """,
    "author": "Lachlan Alexander",
    "date last modified": "10-Sep-2024",
    "change log": "Added variables instantiation for flow rates, delays, and movement speeds for dealing with viscosity."
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

num_reps = 4  # Rows of duplicate samples
num_samples = 8  # Columns of the variable samples
pause_time = 30  # Time robot pauses between dispenses


# Begin protocol
def run(protocol: protocol_api.ProtocolContext):
    # Load labware
    r_tipracks = [
        protocol.load_labware(R_PIP_TIPRACK_LOADNAME, slot) for slot in R_PIP_TIPRACK_SLOTS
    ]
    l_tipracks = [
        protocol.load_labware(L_PIP_TIPRACK_LOADNAME, slot) for slot in L_PIP_TIPRACK_SLOTS
    ]
    reservoir = protocol.load_labware(RESERVOIR_LOADNAME, RESERVOIR_SLOTS[0])
    plate = protocol.load_labware(WELL_PLATE_LOADNAME, WELL_PLATE_SLOTS[0])

    # Load pipettes
    right_pipette = protocol.load_instrument(PIPETTE_R_NAME, "right", tip_racks=r_tipracks)
    left_pipette = protocol.load_instrument(PIPETTE_L_NAME, "left", tip_racks=l_tipracks)

    # Set flow rates for working with DMSO or whichever solvent being used
    right_pipette.flow_rate.aspirate = 41.175
    right_pipette.flow_rate.dispense = 19.215
    right_pipette.flow_rate.blow_out = 5
    withdrawal_speed = 10
    aspiration_delay = 2
    dispense_delay = 2

    # Distribute analyte. Each column is a sample. Rows going down a column are repetitions of that sample.
    for i in range(num_samples):
        right_pipette.pick_up_tip()
        for row in range(num_reps):
            right_pipette.aspirate(300, reservoir.wells()[5])
            protocol.delay(seconds=aspiration_delay)
            right_pipette.move_to(reservoir.wells()[5].top(), speed=withdrawal_speed)
            right_pipette.dispense(300, plate.columns()[i][row])
            protocol.delay(seconds=dispense_delay)
            right_pipette.blow_out()
            right_pipette.move_to(plate.columns()[i][row].top(), speed=withdrawal_speed)
            right_pipette.move_to(reservoir.wells()[5].top())
        right_pipette.drop_tip()

        # Pause protocol for the specified time
        protocol.delay(minutes=0, seconds=5, msg="Tests that the robot pauses at the right time!")

    right_pipette.flow_rate.blow_out = 80  # set to default blow out rate

    # Distribute time zero sample to column 11
    right_pipette.pick_up_tip()
    for row in range(num_reps):
        right_pipette.aspirate(300, reservoir.wells()[5])
        protocol.delay(seconds=aspiration_delay)
        right_pipette.move_to(reservoir.wells()[5].top(), speed=withdrawal_speed)
        right_pipette.dispense(300, plate.columns()[-2][row])
        protocol.delay(seconds=dispense_delay)
        right_pipette.blow_out()
        right_pipette.move_to(plate.columns()[-2][row].top(), speed=withdrawal_speed)
        right_pipette.move_to(reservoir.wells()[5].top())
    right_pipette.drop_tip()

    # Distribute sample blank (solvent only) to column 12
    right_pipette.pick_up_tip()
    for row in range(num_reps):
        right_pipette.aspirate(300, reservoir.wells()[5])
        protocol.delay(seconds=aspiration_delay)
        right_pipette.move_to(reservoir.wells()[5].top(), speed=withdrawal_speed)
        right_pipette.dispense(300, plate.columns()[-1][row])
        protocol.delay(seconds=dispense_delay)
        right_pipette.blow_out()
        right_pipette.move_to(plate.columns()[-1][row].top(), speed=withdrawal_speed)
        right_pipette.move_to(reservoir.wells()[5].top())
    right_pipette.drop_tip()
