"""
combine_gals.py
Author: Connor Hainje
Adapted from Mariangela Lisanti's GlueGalaxies.ipynb

Sample usage:
python combine_gals.py mw_dir sgr_dir out_dir

Description:
Combines the last snapshot from each of two galaxies' GIZMO simulations into a
single HDF5 file to be used as an initial conditions file for GIZMO simulation of
the merged galaxies. Places galaxy 1 (MW) at rest at the origin and galaxy 2 (Sgr)
at the point (125, 0, 0) with an initial velocity of (-10, 0, 70). (These values are
taken from Dierickx and Loeb 2016.)

Notes:
Requires the following directory structure. (Let `dir` be the given directory.)
mw_dir/output/  <- contains the MW GIZMO output snapshots
sgr_dir/output/ <- contains the Sgr GIZMO output snapshots
out_dir needs no special structure.

Also note that the implementation is robust to the presence of an additional `/`
at the end of the given directory; i.e. path/to/dir and path/to/dir/ both work.
"""

import numpy as np
import matplotlib.pyplot as plt

import h5py
import os
from glob import glob

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('mw_dir',  type=str, help="Directory for equilibrated MW simulation.")
ap.add_argument('sgr_dir', type=str, help="Directory for equilibrated Sgr simulation.")
ap.add_argument('out_dir', type=str, help="Output directory.")
ap.add_argument('--verbose', action='store_true', help="Enable verbose output.")
args = ap.parse_args()

trim_slash = lambda s: s[:-1] if s[-1] == '/' else s

mw_dir = trim_slash(args.mw_dir)
sgr_dir = trim_slash(args.sgr_dir)
out_dir = trim_slash(args.out_dir)

mw_snap  = sorted(glob(f'{mw_dir}/output/snapshot_*.hdf5'))[-1]
sgr_snap = sorted(glob(f'{sgr_dir}/output/snapshot_*.hdf5'))[-1]
output_filename = f'{out_dir}/ic.hdf5'

if args.verbose:
    print(f'MW snapshot:  {mw_snap}')
    print(f'Sgr snapshot: {sgr_snap}')
    print(f'Output file:  {output_filename}')

g1 = h5py.File(mw_snap, 'r')
g2 = h5py.File(sgr_snap, 'r')

# Define initial conditions
g1_xyz_start  = np.array([0, 0, 0])
g1_vxyz_start = np.array([0, 0, 0])
g2_xyz_start  = np.array([125, 0, 0])
g2_vxyz_start = np.array([-10, 0, 70])

# Shift gal1 and gal2 to their specified initial position and velocity
g1_part1_coords = g1['PartType1']['Coordinates'] + g1_xyz_start
g1_part1_vels = g1['PartType1']['Velocities'] + g1_vxyz_start
g1_part1_mass = g1['PartType1']['Masses'][:]
g1_part1_ids = g1['PartType1']['ParticleIDs'][:]

g1_part2_coords = g1['PartType2']['Coordinates'] + g1_xyz_start
g1_part2_vels = g1['PartType2']['Velocities'] + g1_vxyz_start
g1_part2_mass = g1['PartType2']['Masses'][:]
g1_part2_ids = g1['PartType2']['ParticleIDs'][:]

g2_part1_coords = g2['PartType1']['Coordinates'] + g2_xyz_start
g2_part1_vels = g2['PartType1']['Velocities'] + g2_vxyz_start
g2_part1_mass = g2['PartType1']['Masses'][:]
g2_part1_ids = g2['PartType1']['ParticleIDs'][:]

g2_part2_coords = g2['PartType2']['Coordinates'] + g2_xyz_start
g2_part2_vels = g2['PartType2']['Velocities'] + g2_vxyz_start
g2_part2_mass = g2['PartType2']['Masses'][:]
g2_part2_ids = g2['PartType2']['ParticleIDs'][:]

# Shift gal1 and gal2 to the COM and COV frame
def find_com():
    Mtot = np.sum(g1_part1_mass) + np.sum(g1_part2_mass) + np.sum(g2_part1_mass) +  np.sum(g2_part2_mass)
    R_cm = []
    for i in range(3):
        xyz_cm = np.sum(g1_part1_coords[:,i] * g1_part1_mass)
        xyz_cm += np.sum(g1_part2_coords[:,i] * g1_part2_mass)
        xyz_cm += np.sum(g2_part1_coords[:,i] * g2_part1_mass)
        xyz_cm += np.sum(g2_part2_coords[:,i] * g2_part2_mass)
        xyz_cm /= Mtot
        R_cm.append(xyz_cm)

    return np.array(R_cm)

def find_cov():
    Mtot = np.sum(g1_part1_mass) + np.sum(g1_part2_mass) + np.sum(g2_part1_mass) +  np.sum(g2_part2_mass)
    V_cm = []
    for i in range(3):
        vxyz_cm = np.sum(g1_part1_vels[:,i] * g1_part1_mass)
        vxyz_cm += np.sum(g1_part2_vels[:,i] * g1_part2_mass)
        vxyz_cm += np.sum(g2_part1_vels[:,i] * g2_part1_mass)
        vxyz_cm += np.sum(g2_part2_vels[:,i] * g2_part2_mass)
        vxyz_cm /= Mtot
        V_cm.append(vxyz_cm)

    return np.array(V_cm)

Rcm = find_com()
Vcm = find_cov()
if args.verbose:
    print(f"Center of mass: {Rcm}")
    print(f"Center of velocity: {Vcm}")

g1_part1_coords_cm = g1_part1_coords - Rcm
g1_part2_coords_cm = g1_part2_coords - Rcm

g1_part1_vels_cm = g1_part1_vels - Vcm
g1_part2_vels_cm = g1_part2_vels - Vcm

g2_part1_coords_cm = g2_part1_coords - Rcm
g2_part2_coords_cm = g2_part2_coords - Rcm

g2_part1_vels_cm = g2_part1_vels - Vcm
g2_part2_vels_cm = g2_part2_vels - Vcm

# Join up the files and align Particle IDs
g1_ndm = len(g1_part1_ids)
g1_nstar = len(g1_part2_ids)

g2_ndm = len(g2_part1_ids)
g2_nstar = len(g2_part2_ids)

if args.verbose:
    print(f"Galaxy 1 - N_DM: {g1_ndm}, N_star: {g1_nstar}")
    print(f"Galaxy 2 - N_DM: {g2_ndm}, N_star: {g2_nstar}")

g1_part1_ids_join = g1_part1_ids
g1_part2_ids_join = g1_part2_ids
g2_part1_ids_join = g2_part1_ids + g1_ndm + g1_nstar
g2_part2_ids_join = g2_part2_ids + g1_ndm + g1_nstar

# Concatenate gal1 and gal2 arrays
part1_coords = np.concatenate((g1_part1_coords_cm, g2_part1_coords_cm))
part2_coords = np.concatenate((g1_part2_coords_cm, g2_part2_coords_cm))

part1_vels = np.concatenate((g1_part1_vels_cm, g2_part1_vels_cm))
part2_vels = np.concatenate((g1_part2_vels_cm, g2_part2_vels_cm))

part1_ids = np.concatenate((g1_part1_ids_join, g2_part1_ids_join))
part2_ids = np.concatenate((g1_part2_ids_join, g2_part2_ids_join))

part1_mass = np.concatenate((g1_part1_mass, g2_part1_mass))
part2_mass = np.concatenate((g1_part2_mass, g2_part2_mass))

if os.path.exists(output_filename):
    os.remove(output_filename)
    if args.verbose: print(f'Existing file at {output_filename} removed.')

file = h5py.File(output_filename, 'w')

npart = np.array([0, g1_ndm + g2_ndm, g1_nstar + g2_nstar, 0, 0, 0])

h = file.create_group('Header')
h.attrs['NumPart_ThisFile'] = npart
h.attrs['NumPart_Total'] = npart
h.attrs['NumPart_Total_HighWord'] = 0 * npart

## everything below this point will just be written over
## need to set values for the code to read

h.attrs['MassTable'] = np.zeros(6)
h.attrs['Time'] = 0.0
h.attrs['Redshift'] = 0.0
h.attrs['BoxSize'] = 1.0
h.attrs['NumFilesPerSnapshot'] = 1
h.attrs['Omega0'] = 1.0
h.attrs['OmegaLambda'] = 0.0
h.attrs['HubbleParam'] = 1.0
h.attrs['Flag_Sfr'] = 0
h.attrs['Flag_Cooling'] = 0
h.attrs['Flag_StellarAge'] = 0
h.attrs['Flag_Metals'] = 0
h.attrs['Flag_Feedback'] = 0
h.attrs['Flag_DoublePrecision'] = 0
h.attrs['Flag_IC_Info'] = 0

p1 = file.create_group('PartType1')
p1.create_dataset('Coordinates', data=part1_coords)
p1.create_dataset('Velocities', data=part1_vels)
p1.create_dataset('ParticleIDs', data=part1_ids)
p1.create_dataset('Masses', data=part1_mass)

p2 = file.create_group('PartType2')
p2.create_dataset('Coordinates', data=part2_coords)
p2.create_dataset('Velocities', data=part2_vels)
p2.create_dataset('ParticleIDs', data=part2_ids)
p2.create_dataset('Masses', data=part2_mass)

file.close()
if args.verbose:
    print(f"Output written to: {output_filename}")
