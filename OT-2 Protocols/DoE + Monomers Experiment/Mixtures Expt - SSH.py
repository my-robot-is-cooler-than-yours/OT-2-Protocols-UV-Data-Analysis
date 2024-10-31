from opentrons import protocol_api
import csv
import json

styrene_volumes = []
polystyrene_volumes = []
solvent_volumes = []

csv_path = "/data/user_storage/prd_protocols/Duplicated_Volumes.csv"

with open(csv_path, mode='r', newline="") as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        styrene_volumes.append(float(row['Styrene (uL)']))
        polystyrene_volumes.append(float(row['Polystyrene (uL)']))
        solvent_volumes.append(float(row['Solvent (uL)']))

# Define constants
num_samples = len(styrene_volumes)  # the number of unique samples to be measured
total_volume = 300  # final volume in each well
step_size = 20  # minimum step size
num_factors = 2  # number of variables (styrene, polystyrene)
well_height = 10.9  # mm from top to bottom of well

# Define metadata for protocol
metadata = {
    "apiLevel": "2.19",
    "protocolName": "DOE Mixtures - SSH",
    "description": """
    From CSV input, produces 46 unique samples of varying polymer and monomer concentrations. The first four wells are
    intended to be used as blanks. 
    """,
    "author": "Lachlan Alexander",
    "date last modified": "23-Oct-2024",
    "change log": "Added SSH-friendly labware loading with offsets taken from OT app. "
                  "Added dynamic volume loading from CSV loaded to OT-2 directory."
                  "Added capability to handle dispensing into plate reader labware."
                  "Added building block commands to work with custom plate defs."
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
    # plate_wells = [well for row in plates[0].rows() for well in row]  # Wells in row-major order
    plate_wells = []  # Reset list
    for row1, row2 in zip(plates[0].columns(), plates[1].columns()):
        plate_wells.extend(row1)  # Add wells from the current row of slot 4 definition
        plate_wells.extend(row2)  # Add wells from the current row of slot 7 definition

    sample_wells = plate_wells  # Can be redefined if required

    # Prepare target wells for experimental samples
    target_wells = [well for pair in zip(sample_wells[::2], sample_wells[1::2]) for well in pair][:2 * num_samples]
    target_wells_bottom = [wll.bottom(well_height / 2.5) for wll in target_wells]
    target_wells_top = [wll.top() for wll in target_wells]

    # # Step 2: Add solvent to wells
    # protocol.comment("Adding solvent to wells")
    # right_pipette.pick_up_tip()
    # for well, volume in zip(target_wells_top, solvent_volumes):
    #     right_pipette.aspirate(volume, reservoirs[0].wells()[0])
    #     right_pipette.dispense(volume, well)
    #     right_pipette.blow_out(reservoirs[0].wells()[0].top())
    # right_pipette.drop_tip()
    #
    # # Step 3: Add styrene to wells
    # protocol.comment("Adding styrene to sample wells")
    # right_pipette.pick_up_tip()
    # for well, volume in zip(target_wells_top, styrene_volumes):
    #     right_pipette.aspirate(volume, reservoirs[0].wells()[-1])
    #     right_pipette.dispense(volume, well)
    #     right_pipette.blow_out(reservoirs[0].wells()[-1].top())
    # right_pipette.drop_tip()
    #
    # # Step 4: Add polystyrene to wells
    # protocol.comment("Adding polystyrene to sample wells")
    # right_pipette.pick_up_tip()
    # for well, volume in zip(target_wells_top, polystyrene_volumes):
    #     right_pipette.aspirate(volume, reservoirs[0].wells()[5])
    #     right_pipette.dispense(volume, well)
    #     right_pipette.blow_out(reservoirs[0].wells()[5].top())
    # right_pipette.drop_tip()
    #

    protocol.comment("Adding solvent to wells")
    left_pipette.transfer(
        volume=solvent_volumes,
        source=reservoirs[0].wells()[0],
        dest=target_wells_bottom,
        blow_out=True,
        blowout_location="destination well"
    )

    # Step 3: Distribute styrene
    protocol.comment("Adding styrene to sample wells")
    left_pipette.transfer(
        volume=styrene_volumes,
        source=reservoirs[0].wells()[-1],
        dest=target_wells_bottom,
        new_tip="once",
        blow_out=True,
        blowout_location="destination well"
    )

    # Step 4: Distribute PS
    protocol.comment("Adding polystyrene to sample wells")
    left_pipette.transfer(
        volume=polystyrene_volumes,
        source=reservoirs[0].wells()[5],
        dest=target_wells_bottom,
        new_tip="always",
        mix_after=(2, 150),
        blow_out=True,
        blowout_location="destination well"
    )

    protocol.comment("Protocol Finished")

    # # Step 1: Add solvent to blank wells
    # protocol.comment("Adding solvent to blank wells")
    #
    # right_pipette.transfer(
    #     volume=total_volume,
    #     source=[reservoirs[0].wells()[0],
    #             reservoirs[0].wells()[0],
    #             reservoirs[0].wells()[0],
    #             reservoirs[0].wells()[0]
    #             ],
    #     dest=[plates[0].wells_by_name()[blank_wells[0]],
    #           plates[0].wells_by_name()[blank_wells[1]],
    #           plates[0].wells_by_name()[blank_wells[2]],
    #           plates[0].wells_by_name()[blank_wells[3]]
    #           ],
    #     new_tip="once",
    #     blow_out=True,
    #     blowout_location="source well"
    # )

    # Step 2: Distribute first half of solvent

    # Step 5: Mix
    # ? lol

    #
    # # Step 4: Transfer polystyrene
    # protocol.comment("Adding polystyrene to sample wells")
    # left_pipette.transfer(
    #     volume=polystyrene_volumes,
    #     source=reservoirs[0].wells()[5],
    #     dest=[wll.bottom(well_height / 2) for wll in target_wells],
    #     new_tip="always",
    #     mix_after=(2, 150),
    #     blow_out=True,
    #     blowout_location="destination well"
    # )


    # # Step 2b: Distribute second half of solvent
    # protocol.comment("Adding solvent to sample wells")
    # right_pipette.distribute(
    #     volume=solvent_volumes[len(solvent_volumes) // 2:],
    #     source=reservoirs[0].wells()[0],
    #     dest=target_wells_bottom[len(target_wells_bottom) // 2:],
    #     blow_out=True,
    #     blow_out_location="source well"
    # )

    # # Step 3b: Distribute second half styrene
    # protocol.comment("Adding styrene to sample wells")
    # right_pipette.distribute(
    #     volume=styrene_volumes[len(styrene_volumes) // 2:],
    #     source=[reservoirs[0].wells()[-1]
    #             ],
    #     dest=target_wells_bottom[len(target_wells_bottom) // 2:],
    #     blow_out=True,
    #     blowout_location="source well"
    # )
