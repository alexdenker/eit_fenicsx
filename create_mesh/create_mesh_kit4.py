import warnings

warnings.filterwarnings("ignore")
import gmsh
import numpy as np
import sys


## ARGUMENTS
mesh_type = "dense"  # dense, coarse
show_mesh = True

L = 16
radius = 1.0

if mesh_type == "dense":
    n_in = 16  # Number of vertices on the electrodes.
    n_out = 8  # Number of vertices in gaps.
elif mesh_type == "coarse":
    n_in = 12
    n_out = 4
else:
    raise NotImplementedError


"""
Compute Electrode position. Code adapted from https://github.com/HafemannE/FEIT_CBM34/blob/main/CBM/FEIT_codes/FEIT_onefile.py
"""


class electrodes_position:
    def __init__(self, L, per_cover, rotate, anticlockwise=True):
        if not isinstance(L, int):
            raise ValueError("Number of electrodes must be an integer.")
        if not isinstance(per_cover, float):
            raise ValueError("per_cover must be a float.")
        if not isinstance(rotate, (int, float)):
            raise ValueError("rotate must be a float.")
        if not isinstance(anticlockwise, bool):
            raise ValueError("anticlockwise must be true of false.")
        if per_cover > 1:
            raise ValueError(
                "per_cover must be equal or less than 1. Example (75%): per_cover=0.75 "
            )

        self.rotate = rotate
        self.L = L
        self.per_cover = per_cover
        self.anticlockwise = anticlockwise

        self.position = self.calc_position()

    def calc_position(self):
        """
        Calculate the position of electrodes based on the :class:`electrodes_position` object.

        :returns: list of arrays -- Returns a list with angle initial and final of each electrode.
        """
        size_e = 2 * np.pi / self.L * self.per_cover  # Size electrodes
        size_gap = 2 * np.pi / self.L * (1 - self.per_cover)  # Size gaps
        rotate = self.rotate  # Rotating original solution

        electrodes = []
        for i in range(self.L):
            # Example electrodes=[[0, pi/4], [pi/2, pi]]
            electrode_start = size_e * i + size_gap * i + rotate
            electrode_end = size_e * (i + 1) + size_gap * i + rotate
            electrode_start = ((electrode_start + np.pi) % (2 * np.pi)) - np.pi
            electrode_end = ((electrode_end + np.pi) % (2 * np.pi)) - np.pi

            electrodes.append([electrode_end, electrode_start])

        if not self.anticlockwise:
            electrodes[1:] = electrodes[1:][
                ::-1
            ]  # Keep first electrode and reverse order
        return electrodes


# The electrodes for KIT4 cover 45% of the boundary
rotate = np.pi / 2 - np.pi / L * 0.45
electrodes_obj = electrodes_position(
    L=L, per_cover=0.45, rotate=rotate, anticlockwise=False
)

print(electrodes_obj.position)


electrodes = np.copy(electrodes_obj.position)
rotate = electrodes_obj.rotate
anticlockwise = electrodes_obj.anticlockwise

gmsh.initialize()
gmsh.model.add("Disk")

point_num = L * (n_in + n_out)
tag_list = np.zeros(point_num, dtype=np.int16)
elec_pos_list = []

no_elek_tags_idx = []

electrode_idx = {}
counter = 0
for i in range(L):
    electrode_idx[i] = []
    # Creating vertex on the electrodes.
    theta0, thetai = electrodes[i][0], electrodes[i][1]
    if thetai > theta0:
        # Add 2*pi to the end angle to ensure continuity
        theta0 += 2 * np.pi
    # print("Electrode at: ", theta0, thetai)
    for idx, theta in enumerate(np.linspace(theta0, thetai, n_in)):
        print(idx, n_in)
        print(f"Counter {counter} is electrode point!")

        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        elec_pos_list.append((radius * np.cos(theta), radius * np.sin(theta)))
        tag_list[counter] = gmsh.model.occ.addPoint(
            radius * np.cos(theta), radius * np.sin(theta), 0.0
        )
        # I am not quite sure, but gmshio needs the last part to be already non electrode to produce the correct final regions 
        if idx < (n_in - 1):
            electrode_idx[i].append(counter)
        else:
            print(f"ADD COUNTER {counter} TO NO ELECTRODE")
            no_elek_tags_idx.append(counter)
        counter += 1

    # Selecting the last gap with the first electrode.
    if i < L - 1:
        theta0, thetai = electrodes[i][1], electrodes[i + 1][0]
    else:
        theta0 = electrodes[i][1]
        thetai = (electrodes[0][0] + np.pi) % (2 * np.pi) - np.pi

        # print(theta0, thetai, (( 2*np.pi+rotate + np.pi) % (2*np.pi)) - np.pi)

    # print("Gap from: ", theta0, thetai)
    # Creating vertex on the gaps.
    for theta in np.linspace(theta0, thetai, n_out + 2):
        if theta != theta0 and theta != thetai:
            print(f"Counter {counter} is NO electrode point!")
            tag_list[counter] = gmsh.model.occ.addPoint(
                radius * np.cos(theta), radius * np.sin(theta), 0.0
            )
            elec_pos_list.append((radius * np.cos(theta), radius * np.sin(theta)))
            no_elek_tags_idx.append(counter)
            counter += 1

print("no elek: ", )

# print("counter and point_num: ", counter, point_num)

print(f"MESH POINTS OF FIRST ELECTRODE FOR {mesh_type} MESH")
print(len(electrode_idx[0]))
total_length = 0
for idx in range(len(electrode_idx[0]) - 1):
    print("Intermediate length: ", np.sqrt((elec_pos_list[idx][0] - elec_pos_list[idx+1][0])**2 + (elec_pos_list[idx][1] - elec_pos_list[idx+1][1])**2) )
    total_length += np.sqrt((elec_pos_list[idx][0] - elec_pos_list[idx+1][0])**2 + (elec_pos_list[idx][1] - elec_pos_list[idx+1][1])**2) 
    print(elec_pos_list[idx])

print("TOTAL LENGTH: ", total_length)

#breakpoint()

mesh_size_center = 0.095
cp_distance = 0.1  # 0.1 #0.0065
cp1 = gmsh.model.occ.addPoint(-cp_distance, cp_distance, 0.0, meshSize=mesh_size_center)
cp2 = gmsh.model.occ.addPoint(cp_distance, cp_distance, 0.0, meshSize=mesh_size_center)
cp3 = gmsh.model.occ.addPoint(
    -cp_distance, -cp_distance, 0.0, meshSize=mesh_size_center
)
cp4 = gmsh.model.occ.addPoint(cp_distance, -cp_distance, 0.0, meshSize=mesh_size_center)

for i in range(len(tag_list) - 1):
    gmsh.model.occ.addLine(tag_list[i], tag_list[i + 1])

gmsh.model.occ.addLine(tag_list[-1], tag_list[0])

gmsh.model.occ.synchronize()

loop = gmsh.model.occ.addCurveLoop(tag_list)
surf = gmsh.model.occ.addPlaneSurface([loop])

gmsh.model.occ.synchronize()

gmsh.model.mesh.embed(0, [cp1, cp2, cp3, cp4], 2, surf)

gmsh.model.occ.synchronize()
### Mark subdomains
gmsh.model.addPhysicalGroup(2, [surf], 1, name="domain")

print("tag list: ", tag_list)
for i in range(L):
    print(f"Electrode_idx for {i}: {electrode_idx[i]}")
    print(f"Selected Tag List {i}: {tag_list[electrode_idx[i]]}")

    gmsh.model.addPhysicalGroup(
        1, tag_list[electrode_idx[i]], i + 1, name="Elektrode" + str(i + 1)
    )

gmsh.model.addPhysicalGroup(1, tag_list[no_elek_tags_idx], 0, name="No-Elektrode")


gmsh.model.mesh.generate(2)


gmsh.write(f"data/KIT4_mesh_{mesh_type}.msh")

if show_mesh and "-nopopup" not in sys.argv:
    gmsh.fltk.run()

# close gmsh
gmsh.finalize()
