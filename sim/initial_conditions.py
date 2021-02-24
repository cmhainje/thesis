"""
initial_conditions.py
Connor Hainje
"""

import numpy as np
import os

mass_unit = 2.325e9 # M_sol


class HaloParams():
    def __init__(self, streaming_fraction=0.5, n_particles=1160000, 
        random_integer_seed=-1, center=True, datafile="dbh.dat",
        R_trunc=200.0, sigma_h=4.375, R_h=20.6, d_rtrunc=20.0, alpha=1.0, beta=3.0
    ):
        # in.halo params
        self.streaming_fraction = streaming_fraction
        self.n_particles = n_particles
        self.random_integer_seed = random_integer_seed
        self.center = 1 if center else 0
        self.datafile = datafile

        # in.dbh params
        self.R_trunc  = R_trunc
        self.sigma_h  = sigma_h
        self.R_h      = R_h
        self.d_rtrunc = d_rtrunc
        self.alpha    = alpha
        self.beta     = beta
        

class DiskParams():
    def __init__(self,         
        n_particles=2030000, random_integer_seed=-1, center=True, datafile="dbh.dat",
        M=34.95, R_d=2.7, R_dtrunc=25.0, z_d=0.36, z_trunc=3.0,
        sigma_r=1.0, R_sigma=2.2, n_spline_points=50, n_iterations=10, dummy="psfile", 
    ):
        # in.disk params
        self.n_particles = n_particles
        self.random_integer_seed = random_integer_seed
        self.center = 1 if center else 0
        self.datafile = datafile

        # in.dbh params
        self.M        = M
        self.R_d      = R_d
        self.R_dtrunc = R_dtrunc
        self.z_d      = z_d
        self.z_trunc  = z_trunc

        # in.diskdf params
        self.sigma_r = sigma_r
        self.R_sigma = R_sigma
        self.n_spline_points = n_spline_points
        self.n_iterations = n_iterations
        self.dummy = dummy
       
    
def make_params(M_NFW, c, M_halo, N_halo, R_200, r_H, M_disk, N_disk, b_0, c_0, M_bulge, N_bulge, a, sigma_h_fudge=2.073,
                disk_n_spline_points=50, disk_n_iterations=10):
    r_NFW = R_200 / c # NFW scale radius
    f = lambda c: np.log1p(c) - c / (1+c)
    sigma_h = np.sqrt((M_halo / mass_unit) / (2 * r_NFW * f(c)) * sigma_h_fudge)
    h = HaloParams(n_particles=N_halo, R_trunc=R_200, sigma_h=sigma_h, R_h=r_NFW)

    d = DiskParams(n_particles=N_disk, M=M_disk / mass_unit,
                   R_d=b_0, R_dtrunc=25.0, R_sigma=b_0, z_d=c_0, z_trunc=3.0, 
                   n_spline_points=disk_n_spline_points, n_iterations=disk_n_iterations)
    return h, d


class Writer:
    def __init__(self, halo=None, disk=None, disk2=None, gas=None, bulge=None, folder='.'):
        if halo is None or disk is None:
            raise Exception("`halo` and `disk` may not be None.")
        if gas is not None:
            print('Warning: `gas` not implemented.')
        if bulge is not None:
            print('Warning: `bulge` not implemented.')
            
        self.halo = halo
        self.disk = disk
        self.disk2 = disk2
        self.gas = gas
        self.bulge = bulge
        self.folder = folder
        
    def _write_dbh(self, smbh="n", dr="0.01", nr="50000", lmax="10"):
        
        halo = self.halo
        disk = self.disk
        disk2 = self.disk2
        gas = self.gas
        bulge = self.bulge
        
        with open(f"{self.folder}/in.dbh", "w") as f:
        
            f.write(f"y\n  {halo.R_trunc} {halo.sigma_h} {halo.R_h} {halo.d_rtrunc} {halo.alpha} {halo.beta}\n")
            f.write(f"y\n  {disk.M:f} {disk.R_d} {disk.R_dtrunc} {disk.z_d} {disk.z_trunc} 0. 0.\n")

            if disk2 is not None:
                f.write(f"y\n  {disk2.M} {disk2.R_d} {disk2.R_dtrunc} {disk2.z_d} {disk2.z_trunc} 0. 0.\n")
            else:
                f.write("n\n")

            if gas is not None:
                f.write("n\n")
            else:
                f.write("n\n")

            if bulge is not None:
                f.write("n\n")
            else:
                f.write("n\n")

            f.write(f"{smbh}\n")    # "Use SMBH"
            f.write(f"{dr} {nr}\n") # dr, nr
            f.write(f"{lmax}")

    def _write_halo(self):
        with open(f"{self.folder}/in.halo", "w") as f:
            f.write(f"{self.halo.streaming_fraction}\n")
            f.write(f"{int(self.halo.n_particles)}\n")
            f.write(f"{self.halo.random_integer_seed}\n")
            f.write(f"{self.halo.center}\n")
            f.write(f"{self.halo.datafile}")

    def _write_disk(self):
        with open(f"{self.folder}/in.disk", "w") as f:
            f.write("1\n")
            f.write(f"{int(self.disk.n_particles)}\n")
            f.write(f"{self.disk.random_integer_seed}\n")
            f.write(f"{self.disk.center}\n")
            f.write(f"{self.disk.datafile}")

    def _write_diskdf(self):
        with open(f"{self.folder}/in.diskdf", "w") as f:
            f.write("1\n")
            f.write(f"{self.disk.sigma_r} {self.disk.R_sigma}\n")
            f.write(f"{self.disk.n_spline_points}\n")
            f.write(f"{self.disk.n_iterations}\n")
            f.write(f"{self.disk.dummy}")

    def _write_disk2(self):
        with open(f"{self.folder}/in.disk2", "w") as f:
            f.write("2\n")
            f.write(f"{int(self.disk2.n_particles)}\n")
            f.write(f"{self.disk2.random_integer_seed}\n")
            f.write(f"{self.disk2.center}\n")
            f.write(f"{self.disk2.datafile}")

    def _write_diskdf2(self):
        with open(f"{self.folder}/in.diskdf2", "w") as f:
            f.write("2\n")
            f.write(f"{self.disk2.sigma_r} {self.disk2.R_sigma}\n")
            f.write(f"{self.disk2.n_spline_points}\n")
            f.write(f"{self.disk2.n_iterations}\n")
            f.write(f"{self.disk2.dummy}")
            
    def write(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self._write_dbh()
        self._write_halo()
        self._write_disk()
        self._write_diskdf()
        if self.disk2 is not None:
            self._write_disk2()
            self._write_diskdf2()
        with open(f"{self.folder}/in.gendenspsi", "w") as f:
            f.write("1000 20\n")