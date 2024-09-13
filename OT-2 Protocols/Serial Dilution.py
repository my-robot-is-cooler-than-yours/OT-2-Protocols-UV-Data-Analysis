from opentrons import protocol_api

# Define metadata for protocol
metadata = {
    "apiLevel": "2.19",
    "protocolName": "Serial dilution 1 in 2",
    "description": """
    Dilutes 175 uL of analyte to 350 uL --> 175 mL total volume in each well. 
    First column is pure analyte, final column is blank.
    Number of replications to be carried out is customisable. 
    """,
    "author": "Lachlan Alexander"
}

# Define variables for protocol
PIPETTE_R_NAME: str = 'p1000_single_gen2'
PIPETTE_L_NAME: str = 'p300_single_gen2'

R_PIP_TIPRACK_SLOTS: list = [7]
R_PIP_TIPRACK_LOADNAME: str = 'opentrons_96_tiprack_1000ul'

L_PIP_TIPRACK_SLOTS: list = [1, 2, 8]
L_PIP_TIPRACK_LOADNAME: str = 'opentrons_96_tiprack_300ul'

RESERVOIR_SLOTS: list = [4, 5, 6]
RESERVOIR_LOADNAME: str = 'nest_12_reservoir_15ml'

WELL_PLATE_SLOTS: list = [3]
WELL_PLATE_LOADNAME: str = 'greiner_96_wellplate_300ul'

num_rows = 4 # the number of rows / repetitions of the dilution to be carried out


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

    # Distribute analyte to first column
    right_pipette.distribute(
        volume=350,
        source=reservoirs[1].wells()[3],
        dest=plates[0].columns()[0][:num_rows],
        blow_out=True,
        blowout_location="source well"
    )

    # Distribute 175 uL diluent to columns 2-12, col 12 is blank
    for i in range(num_rows):
        right_pipette.distribute(
            volume=175,
            source=reservoirs[2].wells()[0],
            dest=plates[0].rows()[i][1:],
            blow_out=True,
            blowout_location="source well"
        )

    # Dilute 175 uL from columns 1 --> 2 --> ... --> 11
    for i in range(num_rows):
        row = plates[0].rows()[i]
        left_pipette.transfer(
            volume=175,
            source=row[0:10],
            dest=row[1:11],
            new_tip="once",
            mix_after=(3, 150),
            blow_out=True,
            blowout_location="destination well"
        )

    # Trash 175 uL from the second last column so every well on the plate is 175 uL in volume
    left_pipette.transfer(
        volume=175,
        source=plates[0].columns()[-2][:num_rows],
        dest=reservoirs[1].wells()[-2],
        new_tip="once",
        blow_out=True,
        blowout_location="destination well"
    )
