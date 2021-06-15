# Experimental setup

As stated previously, the core work of this thesis is performing N-body simulations of the infall of Sgr in the style of Dierickx et al. (todo Dierickx et al. 2017), varying the initial conditions and dark matter model to determine the resulting impacts on the evolution of the Sgr tidal debris stream. As such, we provide an overview of the simulations we performed in this section.

# Pipeline and parameters

To begin our experimental pipeline, we first generate the initial distributions of stellar and dark matter particles using a package called GalactICS (todo GalactICS). Each galaxy is model using a stellar disk and dark matter halo. The halo follows a Navarro-Frenk-White (NFW) profile, as given in Equation (todo reference). The disk follows a Sersic profile.
todo confirm GalactICS uses sersic for disk
todo look up and add equation for Sersic profile

Both of these distributions are subject to truncation beyond a certain radius, $r_t$, with truncation width $dr_t$. The truncation function is given by 
$$ C(r; r_t, dr_t) = \frac{1}{2} \text{erfc} \left( \frac{r - r_t}{\sqrt{2} dr_t} \right). $$
The distributions, including the truncation parameter, can be seen in Figure todo.
todo make plots

Bundled with GalactICS is a subpackage called GadgetConverters, which provides a pipeline for converting the native output of GalactICS into a binary compatible with GADGET and derivative software. In this work, we use GIZMO (todo GIZMO), derived from GADGET-2.

\begin{table}
\begin{tabular}{c}
Parameters
\end{tabular}
\caption{%
    Parameters for the initial Milky Way and Sgr galaxies in our full simulation. These values are in large part taken from the work of (todo Dierickx et al. 2017).
}
\label{tab:sim_params}
\end{table}

The parameters that were used for our simulations were largely taken from the work of Dierickx et al. (todo Dierickx et al. 2017). They are summarized in Table (todo). While they used Hernquist profiles for their halos, they provided the parameters for approximately equivalent NFW distributions. See Figure (todo) for a comparison between the Dierickx Hernquist profiles and the NFW profiles used in our work. We also note that they included a Hernquist bulge in their stellar profile which we have omitted.

todo if GalactICS really does use Sersic, provide a comparison between Sersic and exponential

The experiments performed herein were performed using GIZMO version (todo find out) on Princeton Research Computing’s Della cluster. This cluster is an Intel cluster with $\geq$ 20 cores per node and $\geq$ 128 GB memory per node (cite prc website). Our simulations often required several dozen gigabytes of RAM and typically split the computation over many (todo how many) cores.

# Equilibration

After generating the initial particle distributions for each galaxy, we evolved each one forward in time for a several Gyr to allow it to equilibrate. For our runs involving SIDM, we perform this equilibration run with SIDM microphysics turned on, allowing for the creation of a core in the central region of the dark matter halo.

For each galaxy, we begin with the parameters discussed in the previous section and perform two equilibration runs: one using CDM microphysics and one using SIDM microphysics with a cross section of $\sigma / m = 10$ cm$^2$/g. In this study, we choose to use a somewhat high cross section in order to exaggerate any differences that may appear because of the presence of self-interaction. We note that future studies should consider a range of cross sections.

For the Milky Way equilibrations—both CDM and SIDM—we only evolve the galaxy forward for 2 Gyr, writing time stamps approximately every 0.1 Gyr. This is because we expect the initial distribution to be relatively close to equilibrium, especially when considering such a large galaxy. The resulting evolution of the mass density profiles are shown in Figure todo.

todo add figure of mw eq mass density evolution
todo add discussion of mw eq mass density evolution
todo try to apply SIDM analytic description here

For the Sgr equilibrations, however, we evolved the galaxy much farther forward in time: approximately 10 Gyr for the CDM case and 20 Gyr for SIDM. These evolution times do not correspond to a physical orbit (especially given that the SIDM case would exceed the lifetime of the Universe). Rather, the initial Sgr distribution was found to be a bit unstable. We also wanted to be absolutely certain that the SIDM case would develop a cored profile. The evolution of the resulting Sgr mass profiles is shown in Figure todo.

todo add figure of sgr eq mass density evolution
todo add discussion of sgr eq mass density evolution
todo try to apply SIDM analytic description here

more discussion

With the equilibrated MW and Sgr galaxies in both the cuspy and cored régimes, we combine them to give us two initial conditions for mergers: cuspy and cored. The Milky Way is left at its position from the equilibration run, as its center of mass will be close to the origin and its net velocity will be close to zero. Sgr is placed such that its center of mass lies at the point $[125, 0, 0]$ and is given an initial velocity $[-10,0,70]$. These values correspond to the best fit values found in (todo Dierickx et al. 2017). 