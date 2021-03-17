"""
mw.py
Connor Hainje

Defines parameters for a Milky Way galaxy.
Values from Dierickx & Loeb 2017.

For use with create.ipynb.
"""

M_NFW = 1e12
c = 10
M_halo = 1.25e12
N_halo = 1.16e6
R_200 = 206
r_H = 38.35 # Hernquist scale radius

M_disk = 0.065 * M_halo
N_disk = 2.03e6

b_0 = 3.5        # disk scale length
c_0 = 0.15 * b_0 # disk scale height

M_bulge = 0.01 * M_halo
N_bulge = 3.125e5
a = 0.2 * b_0    # bulge scale length
sigma_h_fudge = 2.340