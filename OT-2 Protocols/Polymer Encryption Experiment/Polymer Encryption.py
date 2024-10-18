from opentrons import protocol_api

# Define constants
total_volume = 350  # final volume in each well
step_size = 20  # minimum step size
well_height = 10.9  # mm from top to bottom of well

# Define metadata for protocol
metadata = {
    "apiLevel": "2.19",
    "protocolName": "Polymer Mixtures Encryption",
    "description": """
    Used to encode a word into mixtures of polymers. Per mixture, five volumes are dispensed in line with 
    the encryption principle described in Vrijsen et al (2020).
    """,
    "author": "Lachlan Alexander",
    "date last modified": "06-Oct-2024",
    "change log": "06-Oct-2024 Created Protocol."
                  "07-Oct-2024 Expanded to auto encode volumes and reservoir source wells. Added liquid handling steps."
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
WELL_PLATE_LOADNAME: str = 'biorad_96_wellplate_200ul_pcr'  # simulation only, remove for actual protocol


# Begin protocol


def create_encoding():
    characters = "abcdefghijklmnopqrstuvwxyz0123456789"  # 36 characters
    encoding_dict = {}
    count = 0

    for level in range(1, 7):  # 6 levels
        for option in range(1, 7):  # 6 options per level
            if count < len(characters):
                encoding_dict[characters[count]] = (level, option)
                count += 1

    return encoding_dict


def encode_word(word):
    encoding_dict = create_encoding()
    encoded_word = []

    for letter in word.lower():
        if letter in encoding_dict:
            encoded_word.append(encoding_dict[letter])
        else:
            # Raise an exception for unsupported characters
            raise ValueError(f"Unsupported character encountered: '{letter}'. Please enter valid letters only.")

    return encoded_word


# Example
word = "STAUDINGER"
encoded = encode_word(word)
samples_required = len(encoded) // 2

print(f"Encoded word for {word}: \n--> {encoded}")
print(f"Samples required to encode word: \n--> {samples_required}")

master_volume = 42.68

volume_dict = {
    1: master_volume * 0.45,
    2: master_volume * 0.75,
    3: master_volume,
    4: master_volume * 1.20,
    5: master_volume * 1.50,
    6: master_volume * 1.80,
}

# Initialize dispense_volumes as an empty list
dispense_volumes = []

# Iterate over pairs of consecutive encoded tuples
for i in range(0, len(encoded), 2):
    if i + 1 < len(encoded):  # Ensure there's a pair to concatenate
        # Get two consecutive tuples
        item1 = encoded[i]
        item2 = encoded[i + 1]

        # Create a 5-tuple combining the two and adding the master volume as standard
        combined_tuple = (
            master_volume,
            volume_dict[item1[0]],  # First value from the first tuple
            volume_dict[item1[1]],  # Second value from the first tuple
            volume_dict[item2[0]],  # First value from the second tuple
            volume_dict[item2[1]],  # Second value from the second tuple
        )
        # Append the 4-tuple to the dispense_volumes list
        dispense_volumes.append(combined_tuple)

reservoir_locations = [(1, 2, 3, 4, 5)] * samples_required

print(f"Volumes to dispense: \n--> {dispense_volumes}")
print(f"Reservoir wells to draw from: \n--> {reservoir_locations}")


def run(protocol: protocol_api.ProtocolContext):
    # Load labware on deck
    r_tipracks: list = [protocol.load_labware(R_PIP_TIPRACK_LOADNAME, slot) for slot in R_PIP_TIPRACK_SLOTS]
    l_tipracks: list = [protocol.load_labware(L_PIP_TIPRACK_LOADNAME, slot) for slot in L_PIP_TIPRACK_SLOTS]
    reservoirs: list = [protocol.load_labware(RESERVOIR_LOADNAME, slot) for slot in RESERVOIR_SLOTS]
    plates: list = [protocol.load_labware(WELL_PLATE_LOADNAME, slot) for slot in WELL_PLATE_SLOTS]

    # Destination wells on the plate (starting from well 1 for actual samples)
    dest_wells = plates[0].wells()[1:samples_required + 1]

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

    # Begin liquid handling steps
    for sample_volumes, reservoir_wells, dest_well in zip(dispense_volumes, reservoir_locations, dest_wells):
        protocol.comment(f"!!!Dispensing into well {dest_well}!!!")

        # Loop through each volume and its corresponding source well
        for volume, res_well_idx in zip(sample_volumes, reservoir_wells):
            source_well = reservoirs[0].wells()[res_well_idx]  # Get the source well from the reservoir

            protocol.comment(f"!!!Transferring {volume} uL from {source_well} to {dest_well}!!!")

            # Perform the liquid transfer for the given volume and source/destination wells
            left_pipette.transfer(
                volume,
                source_well,  # Source well from the reservoir
                dest_well,  # Destination well
                new_tip="always",  # Always pick up a new tip for each transfer
                mix_after=(3, 100),  # Mix after dispensing
                blow_out=True,  # Blow out the liquid
                blowout_location="destination well"  # Blow out in the destination well
            )


if __name__ == "__main__":
    pass
