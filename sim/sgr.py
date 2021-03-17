"""
sgr.py
Connor Hainje

Defines parameters for a Sagittarius dSph galaxy.
Values from Dierickx & Loeb 2017.

For use with create.ipynb.
"""

M_NFW = 1e10
c = 8
M_halo = 1.3e10
N_halo = 1.17e4
R_200 = 44
r_H = 9.81        # Hernquist scale radius

M_disk = 0.06 * M_halo
N_disk = 1.95e4
b_0 = 0.85        # disk scale length
c_0 = 0.15 * b_0  # disk scale height

M_bulge = 0.04 * M_halo
N_bulge = 1.3e4
a = 0.2 * b_0     # bulge scale length
sigma_h_fudge = 2.113