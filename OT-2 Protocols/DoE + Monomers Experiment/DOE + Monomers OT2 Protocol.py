from opentrons import protocol_api
import csv

# Input raw CSV data for volumes of each liquid to be used. Should be a 92 x 3 matrix containing volume values in uL.
# Volumes 23-Aug-2024
csv_1 = '''
Styrene (µL),Polystyrene (µL),Solvent (µL)
96.6,91.8,111.6
96.6,91.8,111.6
20,114,166
20,114,166
55.8,77.2,167
55.8,77.2,167
20,40.6,239.4
20,40.6,239.4
67.4,101,131.6
67.4,101,131.6
100.4,141.8,57.8
100.4,141.8,57.8
113,24.8,162.2
113,24.8,162.2
139.6,20,140.4
139.6,20,140.4
43.8,80.6,175.6
43.8,80.6,175.6
136.8,57.4,105.8
136.8,57.4,105.8
70.2,62.6,167.2
70.2,62.6,167.2
93.8,20,186.2
93.8,20,186.2
20,37.6,242.4
20,37.6,242.4
20,67.4,212.6
20,67.4,212.6
102,97.4,100.6
102,97.4,100.6
20,70.4,209.6
20,70.4,209.6
149.6,43.4,107
149.6,43.4,107
22.4,27.4,250.2
22.4,27.4,250.2
49.8,128,122.2
49.8,128,122.2
64.4,20,215.6
64.4,20,215.6
36.8,124.2,139
36.8,124.2,139
90.2,60.8,149
90.2,60.8,149
131.6,50.2,118.2
131.6,50.2,118.2
54.2,52.8,193
54.2,52.8,193
29.4,144.6,126
29.4,144.6,126
128.8,20,151.2
128.8,20,151.2
114.8,74,111.2
114.8,74,111.2
34.4,139.2,126.4
34.4,139.2,126.4
29,137,134
29,137,134
88,114.6,97.4
88,114.6,97.4
122,84.2,93.8
122,84.2,93.8
118.6,108.6,72.8
118.6,108.6,72.8
146.4,106.8,46.8
146.4,106.8,46.8
39.6,20,240.4
39.6,20,240.4
20,47.2,232.8
20,47.2,232.8
124.8,118.6,56.6
124.8,118.6,56.6
60,122.6,117.4
60,122.6,117.4
45.6,131.6,122.8
45.6,131.6,122.8
79,30.8,190.2
79,30.8,190.2
106.6,148.8,44.6
106.6,148.8,44.6
78.2,88,133.8
78.2,88,133.8
140.6,20,139.4
140.6,20,139.4
74,34.2,191.8
74,34.2,191.8
25.4,88.8,185.8
25.4,88.8,185.8
110.2,20,169.8
110.2,20,169.8
82.2,101.6,116.2
82.2,101.6,116.2
'''

# Volumes 27-Aug-2024
csv_2 = '''
Styrene (µL),Polystyrene (µL),Solvent (µL)
88.60,35.60,175.80
88.60,35.60,175.80
125.60,102.00,72.40
125.60,102.00,72.40
80.20,32.00,187.80
80.20,32.00,187.80
83.00,20.00,197.00
83.00,20.00,197.00
49.80,42.60,207.60
49.80,42.60,207.60
99.20,133.00,67.80
99.20,133.00,67.80
47.20,114.00,138.80
47.20,114.00,138.80
71.60,23.20,205.20
71.60,23.20,205.20
37.80,20.00,242.20
37.80,20.00,242.20
116.00,20.00,164.00
116.00,20.00,164.00
85.80,105.80,108.40
85.80,105.80,108.40
76.20,145.20,78.60
76.20,145.20,78.60
59.20,140.60,100.20
59.20,140.60,100.20
123.40,59.80,116.80
123.40,59.80,116.80
20.00,91.40,188.60
20.00,91.40,188.60
34.80,133.80,131.40
34.80,133.80,131.40
20.00,78.80,201.20
20.00,78.80,201.20
134.40,121.40,44.20
134.40,121.40,44.20
147.20,66.60,86.20
147.20,66.60,86.20
20.00,149.80,130.20
20.00,149.80,130.20
106.80,20.00,173.20
106.80,20.00,173.20
130.40,87.60,82.00
130.40,87.60,82.00
118.40,54.80,126.80
118.40,54.80,126.80
20.00,41.80,238.20
20.00,41.80,238.20
138.20,20.00,141.80
138.20,20.00,141.80
42.40,90.00,167.60
42.40,90.00,167.60
39.60,73.60,186.80
39.60,73.60,186.80
27.80,116.60,155.60
27.80,116.60,155.60
54.60,20.00,225.40
54.60,20.00,225.40
71.80,46.40,181.80
71.80,46.40,181.80
20.00,70.00,210.00
20.00,70.00,210.00
111.40,127.80,60.80
111.40,127.80,60.80
143.80,56.60,99.60
143.80,56.60,99.60
20.00,65.20,214.80
20.00,65.20,214.80
110.60,84.60,104.80
110.60,84.60,104.80
95.00,75.20,129.80
95.00,75.20,129.80
55.60,20.00,224.40
55.60,20.00,224.40
63.60,118.80,117.60
63.60,118.80,117.60
66.60,28.40,205.00
66.60,28.40,205.00
30.60,138.80,130.60
30.60,138.80,130.60
143.40,126.40,30.20
143.40,126.40,30.20
133.60,97.40,69.00
133.60,97.40,69.00
93.00,39.00,168.00
93.00,39.00,168.00
20.80,110.60,168.60
20.80,110.60,168.60
24.40,99.40,176.20
24.40,99.40,176.20
103.60,50.00,146.40
103.60,50.00,146.40
'''

# Volumes 18-Sep-2024, note only two columns
csv_3 = '''
Styrene (µL),Polystyrene (µL)
86.000,214.000
86.000,214.000
232.200,67.800
232.200,67.800
184.800,115.200
184.800,115.200
263.600,36.400
263.600,36.400
249.000,51.000
249.000,51.000
89.600,210.400
89.600,210.400
153.400,146.600
153.400,146.600
150.000,150.000
150.000,150.000
128.600,171.400
128.600,171.400
27.200,272.800
27.200,272.800
179.200,120.800
179.200,120.800
154.200,145.800
154.200,145.800
175.200,124.800
175.200,124.800
145.600,154.400
145.600,154.400
138.400,161.600
138.400,161.600
164.200,135.800
164.200,135.800
174.800,125.200
174.800,125.200
83.800,216.200
83.800,216.200
272.000,28.000
272.000,28.000
170.200,129.800
170.200,129.800
133.000,167.000
133.000,167.000
34.600,265.400
34.600,265.400
153.600,146.400
153.600,146.400
241.000,59.000
241.000,59.000
248.400,51.600
248.400,51.600
135.600,164.400
135.600,164.400
32.000,268.000
32.000,268.000
151.000,149.000
151.000,149.000
195.000,105.000
195.000,105.000
244.800,55.200
244.800,55.200
185.400,114.600
185.400,114.600
20.400,279.600
20.400,279.600
21.800,278.200
21.800,278.200
212.600,87.400
212.600,87.400
64.600,235.400
64.600,235.400
166.600,133.400
166.600,133.400
220.800,79.200
220.800,79.200
183.400,116.600
183.400,116.600
279.000,21.000
279.000,21.000
69.800,230.200
69.800,230.200
68.200,231.800
68.200,231.800
54.600,245.400
54.600,245.400
62.600,237.400
62.600,237.400
113.400,186.600
113.400,186.600
209.200,90.800
209.200,90.800
250.000,50.000
250.000,50.000
'''

# Volumes 23-Sep-2024
csv_raw = """
Styrene (µL),Polystyrene (µL)
213,87
213,87
257.2,42.8
257.2,42.8
60.6,239.4
60.6,239.4
165.8,134.2
165.8,134.2
263.6,36.4
263.6,36.4
89.2,210.8
89.2,210.8
93.8,206.2
93.8,206.2
238.4,61.6
238.4,61.6
113.6,186.4
113.6,186.4
137,163
137,163
192.4,107.6
192.4,107.6
193,107
193,107
109.2,190.8
109.2,190.8
261.6,38.4
261.6,38.4
114,186
114,186
157.8,142.2
157.8,142.2
142.6,157.4
142.6,157.4
194.8,105.2
194.8,105.2
214.2,85.8
214.2,85.8
220.8,79.2
220.8,79.2
135.4,164.6
135.4,164.6
238.2,61.8
238.2,61.8
233.2,66.8
233.2,66.8
67.8,232.2
67.8,232.2
22.2,277.8
22.2,277.8
254.8,45.2
254.8,45.2
182.2,117.8
182.2,117.8
208.2,91.8
208.2,91.8
201.6,98.4
201.6,98.4
33.8,266.2
33.8,266.2
27.8,272.2
27.8,272.2
75.4,224.6
75.4,224.6
107.8,192.2
107.8,192.2
157.4,142.6
157.4,142.6
64.4,235.6
64.4,235.6
264,36
264,36
151,149
151,149
162,138
162,138
41,259
41,259
187.8,112.2
187.8,112.2
118,182
118,182
68.6,231.4
68.6,231.4
161.4,138.6
161.4,138.6
124.4,175.6
124.4,175.6
139.4,160.6
139.4,160.6
68,232
68,232
"""

# Parse the CSV data
csv_data = csv_raw.strip().splitlines()
csv_reader = csv.DictReader(csv_data)

styrene_volumes = []
polystyrene_volumes = []
solvent_volumes = []

for row in csv_reader:
    styrene_volumes.append(float(row['Styrene (µL)']))
    polystyrene_volumes.append(float(row['Polystyrene (µL)']))
    # solvent_volumes.append(float(row['Solvent (µL)']))

# Define constants
num_samples = len(styrene_volumes)  # the number of unique samples to be measured
total_volume = 300  # final volume in each well
step_size = 20  # minimum step size
num_factors = 2  # number of variables (styrene, polystyrene)
well_height = 10.9  # mm from top to bottom of well

# Define metadata for protocol
metadata = {
    "apiLevel": "2.19",
    "protocolName": "DOE Monomer/Polymer Mixtures",
    "description": """
    From CSV input, produces 46 unique samples of varying polymer and monomer concentrations. The first four wells are
    intended to be used as blanks. 
    """,
    "author": "Lachlan Alexander",
    "date last modified": "23-Sep-2024",
    "change log": "23-Sep-2024 Added new volumes. \n\n Test"
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
# WELL_PLATE_LOADNAME: str = 'biorad_96_wellplate_200ul_pcr' # simulation only, remove for actual protocol


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

    # Begin liquid handling steps

    # Prepare well positions
    blank_wells = ['A1', 'B1', 'C1', 'D1']
    sample_wells = plates[0].wells()[4:]  # Starting from the 5th well

    # Prepare target wells for experimental samples
    target_wells = [well for pair in zip(sample_wells[::2], sample_wells[1::2]) for well in pair][:2 * num_samples]
    target_wells_bottom = [wll.bottom(well_height / 2) for wll in target_wells]

    # Step 1: Add solvent to blank wells
    right_pipette.transfer(
        volume=total_volume,
        source=[reservoirs[0].wells()[0],
                reservoirs[0].wells()[0],
                reservoirs[0].wells()[0],
                reservoirs[0].wells()[0]
                ],
        dest=[plates[0].wells_by_name()[blank_wells[0]],
              plates[0].wells_by_name()[blank_wells[1]],
              plates[0].wells_by_name()[blank_wells[2]],
              plates[0].wells_by_name()[blank_wells[3]]
              ],
        new_tip="once",
        blow_out=True,
        blowout_location="source well"
    )

    # Step 2a: Distribute first half of solvent
    # right_pipette.distribute(
    #     volume=solvent_volumes[:len(solvent_volumes) // 2],
    #     source=reservoirs[0].wells()[0],
    #     dest=target_wells_bottom[:len(target_wells_bottom) // 2],
    #     blow_out=True,
    #     blow_out_location="source well"
    # )

    # Step 2b: Distribute second half of solvent
    # right_pipette.distribute(
    #     volume=solvent_volumes[len(solvent_volumes) // 2:],
    #     source=reservoirs[0].wells()[0],
    #     dest=target_wells_bottom[len(target_wells_bottom) // 2:],
    #     blow_out=True,
    #     blow_out_location="source well"
    # )

    # Step 3a: Distribute first half styrene
    right_pipette.distribute(
        volume=styrene_volumes[:len(styrene_volumes) // 2],
        source=[reservoirs[0].wells()[-1]
                ],
        dest=target_wells_bottom[:len(target_wells_bottom) // 2],
        blow_out=True,
        blowout_location="source well"
    )

    # Step 3b: Distribute second half styrene
    right_pipette.distribute(
        volume=styrene_volumes[len(styrene_volumes) // 2:],
        source=[reservoirs[0].wells()[-1]
                ],
        dest=target_wells_bottom[len(target_wells_bottom) // 2:],
        blow_out=True,
        blowout_location="source well"
    )

    # Step 4: Transfer polystyrene
    left_pipette.transfer(
        volume=polystyrene_volumes,
        source=[reservoirs[0].wells()[5],
                reservoirs[0].wells()[5]
                ],
        dest=[wll.bottom(well_height / 2) for wll in target_wells],
        new_tip="always",
        mix_after=(3, 50),
        blow_out=True,
        blowout_location="destination well"
    )
