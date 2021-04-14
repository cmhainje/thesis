"""
nutil.py
Connor Hainje

New utilities!
Merges mergey.py and util.py
Re-made to work with the preprocessed HDF5 datasets
Improved modularization and new features (like Plotly 3D plots!)
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import astropy.coordinates as coord 
import astropy.units as u

from scipy.special import erfc
from matplotlib.gridspec import GridSpec


### MASS DISTRIBUTIONS ###

def truncate(r, r_trunc, dr_trunc):
    return 0.5 * erfc((r - r_trunc) / (np.sqrt(2) * dr_trunc))

def nfw_density_profile(r, sgr=True, trunc=True):
    """Returns the NFW mass density distribution at the specified radii."""
    if sgr:
        M_200, r_200, c = 1e10, 44, 8
    else:
        M_200, r_200, c = 1e12, 206, 10
    f = lambda c: np.log1p(c) - c / (1+c)
    x = r * c / r_200
    rho = M_200 / (4 * np.pi * f(c) * r_200) * c*x / (r * (1+x))**2
    if trunc:
        rho *= truncate(r, r_200, 20)
    return rho

def nfw_enclosed(r, sgr=True):
    """Returns the NFW enclosed mass distribution at the specified radii."""
    if sgr:
        M_200, r_200, c = 1e10, 44, 8
    else:
        M_200, r_200, c = 1e12, 206, 10
    a = r_200 / c
    f = np.log1p(c) - c / (1 + c)
    rho_0 = M_200 / (4 * np.pi * f * a**3)
    x = r / a
    return 4 * np.pi * rho_0 * a**3 * (np.log(1 + x) - x / (1 + x))

def hernquist_density_profile(r, sgr=True):
    """Returns the Hernquist mass density distribution at the 
    specified radii."""
    if sgr:
        M_halo, a = 1.3e10, 9.81
    else:
        M_halo, a = 1.25e12, 38.35
    rho_0 = M_halo / (2 * np.pi * a**3)
    x = r / a
    rho = rho_0 / (x * (1 + x)**3)
    return rho

def hernquist_enclosed(r, sgr=True):
    """Returns the Hernquist enclosed mass distribution at the 
    specified radii."""
    if sgr:
        M_halo, a = 1.3e10, 9.81
    else:
        M_halo, a = 1.25e12, 38.35
    rho_0 = M_halo / (2 * np.pi * a**3)
    x = r / a
    return 2 * np.pi * rho_0 * a**3 * x**2 / (1 + x)**2

def exp_density_profile(r, z=0, sgr=True, trunc=True):
    """Returns the exponential disk mass density distribution at the
    specified radii and z."""
    if sgr:
        R_d, M_disk = 0.85, 7.8e8
    else:
        R_d, M_disk = 3.5, 8.125e10
    h_R = 0.15 * R_d
    
    a = R_d; b = h_R
    coeff = b**2 * M_disk / (4 * np.pi)
    numer = a*r**2 + (a + 3*np.sqrt(z**2+b**2)) * (a + np.sqrt(z**2+b**2))**2
    denom = np.sqrt((r**2 + (a + np.sqrt(z**2+b**2))**2)**5 * (z**2+b**2)**3)
    rho = coeff * numer / denom
    if trunc:
        rho *= truncate(r, 25, 5)
    return rho


### MERGER DATA MANAGER ###

class Merger():

    def __init__(self, filename):

        # The HDF5 dataset
        self.data = h5py.File(filename, 'r')

        # Defines all available snapshot indices
        self.indices = sorted([ 
            int(x[4:]) for x in self.data['mw']['dark'].keys() 
        ])

        # Defines indices to be used in plotting functions, etc.
        self.ilist = self.indices

    def __len__(self):
        return len(self.indices)

    def reset_ilist(self):
        self.ilist = self.indices


    ### DATA UTILS ###
    
    def get_snap(self, galaxy, part, index):
        """Returns the h5py group corresponding to the given galaxy, particle
        type, and snapshot index."""
        return self.data[galaxy][part][f"snap{index}"]

    def get_time(self, index):
        """Returns the time (in Gyr) corresponding to the given snapshot
        index."""
        return self.data['times'][index]
    
    def get_mass(self, galaxy, part):
        """Returns the particle mass in solar masses corresponding to a
        particle in the given galaxy of the given particle type."""
        return self.data[galaxy][part].attrs['mass']
        
    
    ### COMPUTED QUANTITIES ###

    def get_COM(self, galaxy, index, part='star'):
        """Computes the center-of-mass of the given galaxy at the given
        snapshot index for the given particle type. Note that, because all
        particles of a given type in a given galaxy have the same mass, this
        is just the average position of particles of the given type.""" 
        snap = self.get_snap(galaxy, part, index)['pos']
        return np.average(snap, axis=0)
    
    def get_COV(self, galaxy, index, part='star'):
        """Computes the center-of-mass velocity of the given galaxy at the
        given snapshot index for the given particle type. Note that, because
        all particles of a given type in a given galaxy have the same mass,
        this is just the average velocity of particles of the given type."""
        snap = self.get_snap(galaxy, part, index)['vel']
        return np.average(snap, axis=0)

    def get_ICRS(self, galaxy, part, index):
        """Returns the ICRS coordinates of the particles of the given type in
        the given galaxy from the given snapshot index."""
        pos = self.get_snap(galaxy, part, index)['pos']
        vel = self.get_snap(galaxy, part, index)['vel']
        s  = coord.CartesianRepresentation(
            x=pos[:,0], y=pos[:,1], z=pos[:,2], unit=u.kpc
        )
        ds = coord.CartesianDifferential(
            d_x=vel[:,0], d_y=vel[:,1], d_z=vel[:,2], unit=u.km/u.s
        )
        s = s.with_differentials(ds)
        galcen = coord.Galactocentric(s)
        return galcen.transform_to(coord.ICRS)

    def get_COM_in_ICRS(self, galaxy, index, part='star'):
        """Returns the ICRS coordinates of the center-of-mass of the given
        galaxy, particle type (default: 'star'), and snapshot index."""
        com = self.get_COM(galaxy, index, part=part)
        cov = self.get_COV(galaxy, index, part=part)
        com = coord.CartesianRepresentation(
            x=com[0], y=com[1], z=com[2], unit=u.kpc
        )
        cov = coord.CartesianDifferential(
            d_x=cov[0], d_y=cov[1], d_z=cov[2], unit=u.km/u.s
        )
        com = com.with_differentials(cov)
        galcen = coord.Galactocentric(com)
        return galcen.transform_to(coord.ICRS)

    def compute_closest_to_observed(self, topk=6, bound=True):
        """Finds the snapshot indices that correspond to when the simulated
        Sgr is closest to observation. Returns the matrix corresponding to the
        topk closest snapshots in each observable dimension, as well as the
        vector with the topk closest by weighted Euclidean distance."""
        # center of masses
        if bound:
            coms = np.array([
                self.get_bound_COM_in_ICRS(i) for i in self.indices
            ])
        else:
            coms = np.array([self.get_COM_in_ICRS(i) for i in self.indices])

        # actual observed coords of Sgr
        RA, PM_RA_COSDEC = 283.83, -2.54
        DEC, PM_DEC = -29.45, -1.19
        DIST, V_R = 24.8, 139.4
        OBS = np.array([RA, PM_RA_COSDEC, DEC, PM_DEC, DIST, V_R])

        # put it all in a matrix
        vals = np.array([(
            s.ra.degree, s.pm_ra_cosdec.value,
            s.dec.degree, s.pm_dec.value,
            s.distance.value, s.radial_velocity.value
        ) for s in coms])

        # get the closest indices for each dimension
        eachdim = np.argsort(vals - OBS, axis=0)[:topk]
        dst = np.argsort(np.linalg.norm((vals - OBS) / np.array([80, 4, 20, 3, 30, 100]), axis=1, ord=2))[:topk]
        return eachdim, dst
    
    def make_rank_dict(self, part='star'):
        """Makes a rank dictionary for the given particle type. Defaults 
        to star. Rank dictionary contains particle IDs as keys with values
        being the particle's percentile of closeness to the center of mass 
        in the initial snapshot."""
        pos = self.get_snap("sgr", part, 0)['pos']
        pos -= np.average(pos, axis=0)
        distances = [np.sum(np.square(p)) for p in pos]
        rank = np.argsort(distances).argsort()
        pids = self.get_snap("sgr", part, 0)['id']
        if part == 'star':
            self.star_rank = dict([(pid, rk) for pid, rk in zip(pids, rank)])
        else:
            self.dark_rank = dict([(pid, rk) for pid, rk in zip(pids, rank)])
        
    def get_rank(self, index, part='star'):
        """Returns a np.array with the ranks of each particle of the 
        given particle type in the given snapshot index."""
        if part == 'star':
            if not hasattr(self, 'star_rank'):
                raise ArgumentError('Use make_rank_dict to make the ' +
                                    'star rank dictionary before using ' +
                                    'this method.')
            return np.array([self.star_rank[i] for i in self.get_snap('sgr', 'star', index)['id']])
        else:
            if not hasattr(self, 'dark_rank'):
                raise ArgumentError('Use make_rank_dict to make the ' +
                                    'dark rank dictionary before using ' +
                                    'this method.')
            return np.array([self.dark_rank[i] for i in self.get_snap('sgr', 'star', index)['id']])

    def compute_tidal_masks(self, start_radius=25, min_radius=10):

        self.tidal_masks = []
        self.tidal_radii = []

        for index in self.indices:
            pos = self.get_snap('sgr', 'star', index)['pos'][()]
            ids = self.get_snap('sgr', 'star', index)['id'][()]

            if index == 0:
                com = np.average(pos, axis=0)
                dist = np.linalg.norm(pos - com, axis=1)
                mask = dist < start_radius
                mask_dict = dict((k,v) for k,v in zip(ids, mask))
                self.tidal_masks.append(mask_dict)
                self.tidal_radii.append(np.amax(dist[mask]))

            else:
                last_mask = np.array([self.tidal_masks[-1][i] for i in ids])
                bound_pos = pos[last_mask, :]
                com = np.average(bound_pos, axis=0)
                dist = np.linalg.norm(bound_pos - com, axis=1)
                mask = dist < max(min_radius, self.tidal_radii[-1])

                # update mask dict
                mask_dict = self.tidal_masks[-1].copy()
                masked_ids = ids[last_mask]
                for i in range(len(mask)):
                    mask_dict[masked_ids[i]] = mask[i]
                
                self.tidal_masks.append(mask_dict)
                self.tidal_radii.append(np.amax(dist[mask]))
        
        self.tidal_radii = np.array(self.tidal_radii)

    def get_tidal_mask(self, index):
        if not hasattr(self, 'tidal_masks'):
            raise ArgumentError('Use compute_tidal_masks to make the masks ' + 
                                'before using this method.')
        ids = self.get_snap('sgr', 'star', index)['id'][()]
        mask_dict = self.tidal_masks[index]
        mask = np.array([mask_dict[i] for i in ids])
        return mask

    def get_number_unstripped_stars(self, index):
        """Return the number of unstripped Sgr stars (via the tidal masks) in
        the given snapshot."""
        mask = self.get_tidal_mask(index)
        return np.sum(mask)

    def get_bound_COM(self, index):
        """Returns the center-of-mass of the tidally-bound Sgr stars during
        the given snapshot."""
        mask = self.get_tidal_mask(index)
        pos = self.get_snap('sgr', 'star', index)['pos'][()]
        return np.average(pos[mask, :], axis=0)

    def get_bound_COV(self, index):
        """Returns the center-of-mass velocity of the tidally-bound Sgr stars
        during the given snapshot."""
        mask = self.get_tidal_mask(index)
        vel = self.get_snap('sgr', 'star', index)['vel'][()]
        return np.average(vel[mask, :], axis=0)

    def get_bound_COM_in_ICRS(self, index):
        """Returns the ICRS coordinates of the center-of-mass of the
        tidally-bound Sgr stars in the given snapshot."""
        com = self.get_bound_COM(index)
        cov = self.get_bound_COV(index)
        com = coord.CartesianRepresentation(
            x=com[0], y=com[1], z=com[2], unit=u.kpc
        )
        cov = coord.CartesianDifferential(
            d_x=cov[0], d_y=cov[1], d_z=cov[2], unit=u.km/u.s
        )
        com = com.with_differentials(cov)
        galcen = coord.Galactocentric(com)
        return galcen.transform_to(coord.ICRS)


    ### HALO UTILS ###

    def spherical_histogram(self, galaxy, index, r_min, r_max,
        nbins=10, log=True
    ):
        """Calculates the mass histogram of the halo of the given galaxy at
        the given snapshot index. Returns the histogram and centers of each
        radius bin. """
        radii = self.get_snap(galaxy, 'dark', index)['pos.sph'][:,0]
        masses = np.ones_like(radii) * self.get_mass(galaxy, 'dark')
        if log: bin_edges = np.geomspace(r_min, r_max, nbins+1)
        else:   bin_edges = np.linspace(r_min, r_max, nbins+1)
        return np.histogram(radii, bins=bin_edges, weights=masses)

    def spherical_mass_density(self, galaxy, index, r_min, r_max,
        nbins=10
    ): 
        """Calculates the mass density distribution of the halo of galaxy
        `galaxy` at snapshot `index`. Returns the density and the centers 
        of each histogram bin."""
        hist, bin_edges = self.spherical_histogram(
            galaxy, index, r_min, r_max, nbins, log=True
        )
        lower_edges = bin_edges[:-1]
        upper_edges = bin_edges[1:]
        bin_centers = 0.5 * (lower_edges + upper_edges)
        shell_volumes = 4 * np.pi / 3 * (upper_edges**3 - lower_edges**3)
        return hist / shell_volumes, bin_centers
    
    def spherical_enclosed_mass(self, galaxy, index, r_min, r_max,
        nbins=10
    ):
        """Calculates the enclosed mass distribution of the halo of galaxy
        `galaxy` at snapshot `index`. Returns the masses and centers of each
        histogram bin."""
        hist, bin_edges = self.spherical_histogram(
            galaxy, index, r_min, r_max, nbins, log=False
        )
        for i in range(len(hist)):
            hist[i] += hist[i-1]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return hist, bin_centers
    

    ### DISK UTILS ###

    def cylindrical_histogram(self, galaxy, index, r_min, r_max, 
        nbins=10, log=True
    ):
        """Calculates the mass histogram of the disk of the given galaxy at
        the given snapshot index. Returns the histogram and centers of each
        radius bin."""
        radii = self.get_snap(galaxy, 'star', index)['pos.cyl'][:,0]
        masses = np.ones_like(radii) * self.get_mass(galaxy, 'dark')
        if log: bin_edges = np.geomspace(r_min, r_max, nbins+1)
        else:   bin_edges = np.linspace(r_min, r_max, nbins+1)
        return np.histogram(radii, bins=bin_edges, weights=masses)
    
    def cylindrical_mass_density(self, galaxy, index, r_min, r_max, 
        nbins=10
    ):
        """Calculates the mass density distribution as a function of
        cylindrical radius. Returns the density and the centers of each
        histogram bin."""
        hist, bin_edges = self.cylindrical_histogram(
            galaxy, index, r_min, r_max, nbins=nbins, log=True
        )
        lower_edges = bin_edges[:-1]
        upper_edges = bin_edges[1:]
        bin_centers = 0.5 * (lower_edges + upper_edges)
        height = 2 # approximate height of cylindrical shells
        shell_volumes = np.pi * (upper_edges**2 - lower_edges**2) * height
        return hist / shell_volumes, bin_centers
        
    def cylindrical_enclosed_mass(self, galaxy, index, r_min, r_max, 
        nbins=10
    ):
        """Calculates the enclosed mass distribution of the disk of galaxy
        `galaxy` at snapshot `index`. Returns the enclosed mass and center of
        each radius bin."""
        hist, bin_edges = self.cylindrical_histogram(
            galaxy, index, r_min, r_max, nbins, log=False
        )
        for i in range(len(hist)):
            hist[i] += hist[i-1]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return hist, bin_centers


    ### PLOTTING UTILS ###

    def plot_halo_density(self, ax, galaxy,
        nbins=10, r_trunc=True, reference=True, title='',
        xlim=None, ylim=None, xscale='log', yscale='log'
    ):
        """Plots the density of the halo of the given galaxy for the snapshot
        indices specified by the class attribute ilist on the given axis."""
        if xlim is None: xlim = (0.2, 200)
        rainbow = plt.cm.rainbow(np.linspace(0, 1, num=len(self)))
        for i in self.ilist:
            rho, r = self.spherical_mass_density(galaxy, i, xlim[0], xlim[1],
                                                 nbins=nbins)
            ax.plot(r, rho, color=rainbow[i], linewidth=1.5)
        
        if r_trunc:
            ax.axvline(44 if galaxy == 'sgr' else 206, color='grey',
                       linewidth=0.5)

        if reference == True or reference == 'nfw':
            r = np.geomspace(xlim[0], xlim[1], 101)
            ax.plot(r, nfw_density_profile(r, sgr=(galaxy=='sgr')), 'k--')
        elif reference == 'hernquist':
            r = np.geomspace(xlim[0], xlim[1], 101)
            ax.plot(r, hernquist_density_profile(r, sgr=(galaxy=='sgr')), 'k--')
        else:
            r = np.geomspace(xlim[0], xlim[1], 101)
            ax.plot(r, nfw_density_profile(r, sgr=(galaxy=='sgr')), 'k--')
            ax.plot(r, hernquist_density_profile(r, sgr=(galaxy=='sgr')), '--')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel('$R_{gc}$ [kpc]', fontsize=14)
        ax.set_ylabel('Density [M$_\odot$/kpc$^3$]', fontsize=14)
        ax.set_title(title, fontsize=16)

    def plot_halo_enclosed(self, ax, galaxy,
        nbins=10, r_trunc=True, reference=True, title='',
        xlim=None, ylim=None, xscale='linear', yscale='linear'
    ):
        """Plots the enclosed mass distribution of the halo of the given
        galaxy for the snapshot indices specified by the class attribute ilist
        on the given axis."""
        if xlim is None: xlim = (0, 206)
        rainbow = plt.cm.rainbow(np.linspace(0, 1, num=len(self)))
        for i in self.ilist:
            m, r = self.spherical_enclosed_mass(galaxy, i, xlim[0], xlim[1],
                                                nbins=nbins)
            ax.plot(r, m, color=rainbow[i], linewidth=1.5)
        
        if r_trunc:
            ax.axvline(44 if galaxy == 'sgr' else 206, color='grey',
                       linewidth=0.5)

        if reference == True or reference == 'nfw':
            r = np.linspace(xlim[0], xlim[1], 101)
            ax.plot(r, nfw_enclosed(r, sgr=(galaxy=='sgr')), 'k--')
        elif reference == 'hernquist':
            r = np.linspace(xlim[0], xlim[1], 101)
            ax.plot(r, hernquist_enclosed(r, sgr=(galaxy=='sgr')), 'k--')
        else:
            r = np.linspace(xlim[0], xlim[1], 101)
            ax.plot(r, nfw_enclosed(r, sgr=(galaxy=='sgr')), 'k--')
            ax.plot(r, hernquist_enclosed(r, sgr=(galaxy=='sgr')), '--')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel('$R_{gc}$ [kpc]', fontsize=14)
        ax.set_ylabel('Mass [M$_\odot$]', fontsize=14)
        ax.set_title(title, fontsize=16)

    def plot_disk_density(self, ax, galaxy,
        nbins=10, r_trunc=True, reference=True, title='',
        xlim=None, ylim=None, xscale='log', yscale='log'
    ):
        """Plots the density of the disk of the given galaxy for the snapshot
        indices specified by the class attribute ilist on the given axis."""
        if xlim is None: xlim = (0.2, 200)
        rainbow = plt.cm.rainbow(np.linspace(0, 1, num=len(self)))
        for i in self.ilist:
            rho, r = self.cylindrical_mass_density(galaxy, i, xlim[0], xlim[1],
                                                   nbins=nbins)
            ax.plot(r, rho, color=rainbow[i], linewidth=1.5)
        
        if r_trunc:
            ax.axvline(25, color='grey', linewidth=0.5)

        if reference:
            r = np.geomspace(xlim[0], xlim[1], 101)
            ax.plot(r, exp_density_profile(r, sgr=(galaxy=='sgr')), 'k--')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel('$R_{gc}$ [kpc]', fontsize=14)
        ax.set_ylabel('Density [M$_\odot$/kpc$^3$]', fontsize=14)
        ax.set_title(title, fontsize=16)

    def plot_disk_enclosed(self, ax, galaxy,
        nbins=10, r_trunc=True, title='',
        xlim=None, ylim=None, xscale='log', yscale='log'
    ):
        """Plots the enclosed mass distribution of the disk of the given
        galaxy for the snapshot indices specified by the class attribute ilist
        on the given axis."""
        if xlim is None: xlim = (0, 50)
        rainbow = plt.cm.rainbow(np.linspace(0, 1, num=len(self)))
        for i in self.ilist:
            m, r = self.cylindrical_enclosed_mass(galaxy, i, xlim[0], xlim[1],
                                                    nbins=nbins)
            ax.plot(r, m, color=rainbow[i], linewidth=1.5)
        
        if r_trunc:
            ax.axvline(25, color='grey', linewidth=0.5)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel('$R_{gc}$ [kpc]', fontsize=14)
        ax.set_ylabel('Mass [M$_\odot$]', fontsize=14)
        ax.set_title(title, fontsize=16)

    def plot_separation(self, ax, 
        fmt='-', bound=True
    ):
        """Plots the separation between the MW center-of-mass and the
        center-of-mass of the tidally-bound stars in Sgr as a function of
        time."""
        # Compute center of mass for each snapshot
        mw_com = np.array([self.get_COM('mw', i) for i in self.ilist])
        if bound:
            sgr_com = np.array([self.get_bound_COM(i) for i in self.ilist])
        else:
            sgr_com = np.array([self.get_COM(i) for i in self.ilist])

        # Compute distance between galaxies for each snapshot
        dists = np.sqrt(
            (mw_com[:,0] - sgr_com[:,0])**2
            + (mw_com[:,1] - sgr_com[:,1])**2
            + (mw_com[:,2] - sgr_com[:,2])**2
        )

        # Plot separation vs time on axis
        ax.plot(self.get_time(self.ilist), dists, fmt)

    def plot_stars_on_axes(self, ax0, ax1, ax2, index,
        cmap='jet', size=3, near=True, cbar=True
    ):
        """Plots the star particles of the Sgr galaxy in four dimensions:
        right ascension, declination, heliocentric distance, and line-of-sight
        velocity"""
        ax = [ax0, ax1, ax2]
        cmap = plt.get_cmap(cmap)

        s = self.get_ICRS('sgr', 'star', index)

        dist_mask = s.distance.value < 100
        if not near: dist_mask = ~dist_mask

        ra = s.ra.degree[dist_mask]
        dec = s.dec.degree[dist_mask]
        dist = s.distance.value[dist_mask]
        vlos = s.radial_velocity.value[dist_mask]

        sc = ax[0].scatter(ra, dec, s=size, c=vlos, cmap=cmap)
        if cbar: plt.colorbar(sc, ax=ax[0], label='$v_{LOS}$ [km s$^{-1}$]')
        ax[0].set_ylabel('Dec [deg]', fontsize=12)

        sc = ax[1].scatter(ra, vlos, s=size, c=dist, cmap=cmap)
        if cbar: plt.colorbar(sc, ax=ax[1], label='Heliocentric distance [kpc]')
        ax[1].set_ylabel('LOS velocity [km s$^{-1}$]', fontsize=12)

        sc = ax[2].scatter(ra, dist, s=size, c=vlos, cmap=cmap)
        if cbar: plt.colorbar(sc, ax=ax[2], label='$v_{LOS}$ [km s$^{-1}$]')
        ax[2].set_ylabel('Heliocentric distance [kpc]', fontsize=12)

        for a in ax:
            a.set_xlabel('RA [deg]', fontsize=12)
            a.set_xlim((360,0))

        if near:
            ax[0].set_ylim((-80,80))
            ax[1].set_ylim((-600,600))
            ax[2].set_ylim((0,100))

        ax[0].set_title(
            f'$d {"<" if near else ">"} 100$ kpc, '
            f'{self.get_time(index):.2f} Gyr'
        )


    ### PLOTTERS ###

    def plot_mw_densities(self,
        title="",
        halo_xlim=(0.1,250),
        halo_ylim=(1e3, 1e10),
        halo_nbin=15,
        disk_xlim=(0.1,250),
        disk_ylim=(1e3, 1e10),
        disk_nbin=15,
    ):
        """Plots the halo and disk density distributions of the Milky Way and
        returns the figure object."""

        fig, ax = plt.subplots(1, 2, figsize=(13,6), tight_layout=True)

        self.plot_halo_density(
            ax[0], 'mw', title='Halo', 
            xlim=halo_xlim, ylim=halo_ylim, nbins=halo_nbin
        )

        self.plot_disk_density(
            ax[1], 'mw', title='Disk', 
            xlim=disk_xlim, ylim=disk_ylim, nbins=disk_nbin
        )

        for a in ax:
            a.legend(
                [f"{self.get_time(i):.2f} Gyr" for i in self.ilist] +
                ['$r_{trunc}$', 'Reference'], loc='lower left', frameon=False
            )

        fig.suptitle(title, fontsize=16)
        return fig

    def plot_mw_halo(self,
        title="",
        dens_xlim=(0.1,250),
        dens_ylim=(1e3, 1e10),
        dens_nbin=15,
        encl_xlim=(0,210),
        encl_ylim=None,
        encl_nbin=50,
    ):
        fig, ax = plt.subplots(1, 2, figsize=(13,6), tight_layout=True)

        self.plot_halo_density(
            ax[0], 'mw', title='Mass density',
            xlim=dens_xlim, ylim=dens_ylim, nbins=dens_nbin, reference='both'
        )
        self.plot_halo_enclosed(
            ax[1], 'mw', title='Enclosed mass',
            xlim=encl_xlim, ylim=encl_ylim, nbins=encl_nbin, reference='both'
        )

        for a in ax:
            a.legend(
                [f"{self.get_time(i):.2f} Gyr" for i in self.ilist] +
                ['$r_{trunc}$', 'Reference'], frameon=False
            )

        fig.suptitle(title, fontsize=16)
        return fig
    
    def plot_infall_trajectory(self, title=""):
        rainbow = plt.cm.rainbow(np.linspace(0, 1, num=len(self)))

        # Compute center of mass for each snapshot
        mw_com = np.array([self.get_COM('mw', i) for i in self.ilist])
        sgr_com = np.array([self.get_bound_COM(i) for i in self.ilist])

        # Compute distance between galaxies for each snapshot
        dists = np.sqrt(
            (mw_com[:,0] - sgr_com[:,0])**2
            + (mw_com[:,1] - sgr_com[:,1])**2
            + (mw_com[:,2] - sgr_com[:,2])**2
        )

        # Make plot
        fig = plt.figure(constrained_layout=True, figsize=(10,7))
        gs = GridSpec(2, 3, figure=fig)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, :])

        # x vs y
        ax0.plot(mw_com[:,0], mw_com[:,1], 'k*', markersize=10)
        for s, i in zip(sgr_com, self.ilist):
            ax0.plot(s[0], s[1], '.', color=rainbow[i], markersize=8)
        ax0.set_xlabel('$x$ [kpc]', fontsize=14)
        ax0.set_ylabel('$y$ [kpc]', fontsize=14)

        # x vs z
        ax1.plot(mw_com[:,0], mw_com[:,2], 'k*', markersize=10)
        for s, i in zip(sgr_com, self.ilist):
            ax1.plot(s[0], s[2], '.', color=rainbow[i], markersize=8)
        ax1.set_xlabel('$x$ [kpc]', fontsize=14)
        ax1.set_ylabel('$z$ [kpc]', fontsize=14)

        # y vs z
        ax2.plot(mw_com[:,1], mw_com[:,2], 'k*', markersize=10)
        for s, i in zip(sgr_com, self.ilist):
            ax2.plot(s[1], s[2], '.', color=rainbow[i], markersize=8)
        ax2.set_xlabel('$y$ [kpc]', fontsize=14)
        ax2.set_ylabel('$z$ [kpc]', fontsize=14)

        for ax in [ax0, ax1, ax2]:
            ax.set_xlim(-150, 150)
            ax.set_ylim(-150, 150)

        # distance
        for i, d in zip(self.ilist, dists):
            ax3.plot(self.get_time(i), d, 'o', color=rainbow[i])
        ax3.set_xlabel('Time of evolution [Gyr]', fontsize=14)
        ax3.set_ylabel('Relative separation [kpc]', fontsize=14)
        ax3.grid()

        fig.suptitle(title)
        return fig

    def plot_stars(self, index,
        cmap='jet', size=3, near=True, cbar=True, title=""
    ):
        fig, ax = plt.subplots(3, 1, figsize=(8,10), tight_layout=True)
        self.plot_stars_on_axes(*ax, index, cmap=cmap, near=near, cbar=cbar)
        fig.suptitle(title)
        return fig

    def movie_stars(self, fname=None):
        fig, ax = plt.subplots(3, 1, figsize=(8,10), tight_layout=True)

        def init():
            return fig,

        def update(frame):
            for a in ax:
                a.clear()
            self.plot_stars_on_axes(*ax, frame, near=True, cbar=False)
            return fig,

        ani = FuncAnimation(fig, update, frames=self.ilist, 
                            init_func=init, blit=False)
        if fname is not None:
            ani.save(fname, dpi=200)
        plt.close()
        return ani.to_html5_video()

    def plot_obs_coords(self,
        ra_lim=(250, 350),
        pm_ra_lim=(-4, 0),
        dec_lim=(-40, -20),
        pm_dec_lim=(-3, 0),
        dist_lim=(5, 45),
        v_r_lim=(100, 200),
        bound=True
    ):
        fig, axs = plt.subplots(1, 3, figsize=(13, 4), tight_layout=True)
        colors = plt.cm.rainbow(np.linspace(0, 1, num=len(self)))

        def add_observed(ax, x, y, xerr=None, yerr=None):
            ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                    capsize=2.5, color='black', elinewidth=0.75, marker='.', markersize=7)

        # True values
        RA, PM_RA_COSDEC = 283.83, -2.54
        DEC, PM_DEC = -29.45, -1.19
        DIST, V_R = 24.8, 139.4

        add_observed(axs[0], RA, PM_RA_COSDEC, yerr=0.18)   # (RA, μ_α cosδ)
        add_observed(axs[1], DEC, PM_DEC, yerr=0.16)        # (dec, μ_δ)
        add_observed(axs[2], DIST, V_R, xerr=0.8, yerr=0.6) # (D, V_r)

        def add_data(ax, x, y, color):
            ax.plot(x, y, 's', markersize=5, color=color)

        if bound:
            coms = np.array([self.get_bound_COM_in_ICRS(i) for i in self.ilist])
        else:
            coms = np.array([self.get_COM_in_ICRS(i) for i in self.ilist])

        for s, i in zip(coms, self.ilist):
            add_data(axs[0], s.ra.degree, s.pm_ra_cosdec.value,
                     color=colors[i])
            add_data(axs[1], s.dec.degree, s.pm_dec.value,
                     color=colors[i])
            add_data(axs[2], s.distance.value, s.radial_velocity.value,
                     color=colors[i])

        axs[0].set_xlabel('RA [$^\circ$]', fontsize=14)
        axs[0].set_ylabel('$\mu_\\alpha \ \cos\,\delta$ [mas yr$^{-1}$]', fontsize=14)

        axs[1].set_xlabel('Dec [$^\circ$]', fontsize=14)
        axs[1].set_ylabel('$\mu_\delta \,$ [mas yr$^{-1}$]', fontsize=14)

        axs[2].set_xlabel('Heliocentric distance [kpc]', fontsize=14)
        axs[2].set_ylabel('Radial velocity [km s$^{-1}$]', fontsize=14)

        axs[0].set_xlim(ra_lim)
        axs[0].set_ylim(pm_ra_lim)
        axs[1].set_xlim(dec_lim)
        axs[1].set_ylim(pm_dec_lim)
        axs[2].set_xlim(dist_lim)
        axs[2].set_ylim(v_r_lim)

        return fig

    def plot_hex_density(self, index, 
        galaxy="sgr", part="star",
        title_prefix="", gridsize=100, cmap='binary',
        vmin=1, vmax=1e2, cbar=True, tight_layout=False,
        ext=200, mask_bound=False, mask_cmap='Reds', mask_alpha=0.5
    ):
        if mask_bound and (galaxy != "sgr" or part != "star"):
            raise ArgumentError("Can't use the tidal mask with" +
                                "DM particles or the MW data.")

        fig, ax = plt.subplots(1, 3, figsize=(15,5), tight_layout=tight_layout)

        pos = self.get_snap(galaxy, part, index)['pos']

        if mask_bound:
            pos = pos[()]
            masked_pos = pos[self.get_tidal_mask(index),:]

        dims = [(0,1), (0,2), (1,2)]
        dim_names = ['$x$ [kpc]', '$y$ [kpc]', '$z$ [kpc]']

        for a, d in zip(ax, dims):
            hb = a.hexbin(pos[:,d[0]], pos[:,d[1]], gridsize=gridsize,
                bins='log', extent=(-ext, ext, -ext, ext), cmap=cmap,
                vmin=vmin, vmax=vmax)

            if mask_bound:
                a.hexbin(masked_pos[:,d[0]], masked_pos[:,d[1]], 
                    gridsize=gridsize, extent=(-ext, ext, -ext, ext),
                    cmap=mask_cmap, vmin=0, vmax=1, mincnt=1,
                    alpha=mask_alpha)

            a.set_xlabel(dim_names[d[0]], fontsize=14)
            a.set_ylabel(dim_names[d[1]], fontsize=14)

        if cbar:
            fig.subplots_adjust(right=0.95)
            cbar_ax = fig.add_axes([0.975, 0.15, 0.01, 0.7])
            fig.colorbar(hb, cax=cbar_ax)

        fig.suptitle(f"{title_prefix}{', ' if title_prefix != '' else ''}" +
                     f"time: {self.get_time(index):.2f} Gyr", fontsize=14)
        return fig

    def plotly_3D_movie(self, galaxy='sgr', part='star',
        width=1000, height=700, opacity=0.05, show=True,
        mask=False, rank=False
    ):
        """Make an interactive 3D display of the galaxy particles over time
        using Plotly. If mask, colors bound and unbound Sgr stars differently.
        If rank, colors particles by their distance from the initial center of
        mass."""
        snaps = [self.get_snap(galaxy, part, i)['pos'] for i in self.ilist]
        df = pd.concat(
            [pd.DataFrame(data=snap, columns=['x','y','z']) for snap in snaps],
            keys=[self.get_time(i) for i in self.ilist],
            names=['t', 'id']
        )
        df.reset_index(inplace=True)

        if mask:
            if galaxy != 'sgr' or part != 'star':
                raise ArgumentError("Can't use mask when looking at DM " + 
                                    " particles or MW data.")
            if rank:
                print('Ignoring `rank`... using `mask`.')

            unstripped = np.concatenate([
                self.get_tidal_mask(i) for i in self.ilist
            ])
            fig = px.scatter_3d(
                df, x="x", y="y", z="z",
                animation_frame="t", animation_group="id",
                range_x=[-200,200], range_y=[-200,200], range_z=[-200,200],
                color=unstripped, width=width, height=height, opacity=opacity
            )
        
        elif rank:
            if galaxy != 'sgr':
                raise ArgumentError("Can't use rank with MW")
            ranks = np.concatenate([
                self.get_rank(i, part=part) for i in self.ilist
            ])
            fig = px.scatter_3d(
                df, x="x", y="y", z="z", 
                animation_frame="t", animation_group="id",
                range_x=[-200,200], range_y=[-200,200], range_z=[-200,200],
                color=ranks,
                color_continuous_scale=px.colors.sequential.Jet,
                width=width, height=height, opacity=opacity)

        else:
            fig = px.scatter_3d(
                df, x="x", y="y", z="z",
                animation_frame="t", animation_group="id",
                range_x=[-200,200], range_y=[-200,200], range_z=[-200,200],
                width=width, height=height, opacity=opacity
            )

        if show:
            fig.show()
        else:
            return fig



