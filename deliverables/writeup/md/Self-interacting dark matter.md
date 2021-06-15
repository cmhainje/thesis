# Self-interacting dark matter
# Introduction
Self-interacting dark matter (SIDM) was introduced in 2000 by Spergel and Steinhardt (Spergel and Steinhardt 2000). The model was proposed as a solution to the core-cusp and missing satellites problems, as the addition of self-interactions was thought to have three primary effects on the distribution of dark matter.
1. Self-interactions in regions of high density would cause dark matter particles to be unbound, reducing the density of halos primarily in their center region. This would yield a cored density profile rather than a cuspy one, solving the core-cusp problem.
2. It is also expected that interactions would yield a more isotropic velocity dispersion than seen in CDM as well as erase the typical triaxial ellipticity seen in halo shapes. In other words, SIDM halos should be more spherical, a potentially testable prediction.
3. Through the processes of isotropizing the velocity dispersion and reducing density in core regions, it was also expected that substructure would be greatly reduced, lowering the number of dwarf galaxies and thereby solving the missing satellites problem.
However, since the dark matter scattering rate would be naturally dependent on the dark matter density, these effects would only be expected in higher density regions. Toward the outermost radii of halos and on larger scales, the effects of self-interaction would be negligible, preserving the large-scale successes of the $\Lambda$CDM model.

Shortly thereafter began a wave of numerical simulations to test these predictions. The results of these simulations were mixed. Some were focused on the evolution of galaxies and found confirmation of the predicted effects (more strongly spherical shape, cored density profile) like (Burkert 2000, Dave et al. 2001). Other simulations, like cluster and cosmological simulations, seemed to show that SIDM would be inconsistent with observation (Moore et al. 2000, Yoshida et al. 2000a, Yoshida et al. 2000b). 

At the same time, constraints on the allowed self-interaction cross section began being compiled, primarily from clusters. (Meneghetti et al. 2001) used cluster simulations to limit the allowable cross section to mass ratio to $< 0.1$ cm$^2$/g. Similarly, gravitational lensing data from the MS 2137-23 cluster was used by (Miralda-Escude 2002) to limit the cross section to mass ratio to $< 10^{-25.5}$ cm$^2$/GeV, approximately 0.02 cm$^2$/g. These cross sections were far too small to solve the small-scale problems in galaxies, where simulations like (Dave et al. 2001) suggested the necessary cross section to be in the range $10^{-25}$ to $10^{-23}$ cm$^2$/MeV (0.05 to 5 cm$^2$/g). 

However, more recent simulations with higher resolution and better statistics have relaxed many of these constraints significantly, improving the viability of the model and once again sparking interest in self-interactions. 
todo detail these new simulations and why they make sidm attractive

todo add a discussion of observations that indicate that the cross-section is velocity-dependent over large scales
todo add one of those plots that shows it changing slowly for galactic scales
todo add a discussion of what different kinds of astrophysical observations might mean different things about the particle physics properties

# Predicted density profile
todo run through an analytic calculation of core size, etc.
todo Jeans approach!

In 2014, Kaplinghat et al. provided a semi-analytic approach to deriving equilibrium solutions for the dark matter density profile using the Jeans equation. This approach allows for the inclusion of the gravitational potential of both the dark matter and baryons, providing the means to make predictions about the effects of baryons on the dark matter distribution. 

They begin with the Jeans equation, rewritten via Poisson’s equation. They assume constant velocity dispersion $\sigma_0$ and that the dark matter density profile is given by $\rho(\mathbf{r}) = \rho_0 \exp(h(\mathbf{r}))$. Plugging these in yields
$$ \nabla_x^2 h(\mathbf{x}) + (4\pi G_N r_0^2/\sigma_0^2)[\rho_B(\mathbf{x}) + \rho_0 \exp(h(\mathbf{r}))] = 0, $$
where we have introduced a length scale $r_0$, a dimensionless length $x = r / r_0$, and the baryonic density profile $\rho_B$.

As a simple test of this equation, we can consider the case where baryonic matter dominates. Then, the $\exp(h)$ term can be neglected, yielding

todo work through this derivation with a few more steps

The solution in this case can then be written as 
$$ \rho(\mathbf{x}) = \rho_0 \exp[(\Phi_B(0) - \Phi_B(\mathbf{x}))/\sigma_0^2]. $$
The authors then recommend defining the core radius as the radius at which the density is half $\rho_0$. Such a position would give $h(\mathbf{r}_c) = -\ln 2$, or 
$$ \Phi(0) - \Phi(\mathbf{r}_c) = -\sigma_0^2 \ln 2. $$
Thus, the core size would depend only on the baryonic potential in the case where it dominates, which follows from these assumptions but stands in marked contrast to observation, where the baryonic contribution does not dominate.

todo finish this discussion

# Particle physics models
Given that very little is known conclusively about the particle physics nature of dark matter, the introduction of the possibility of self-interactions makes way for a wealth of rich new theories. We will cover a few of the most popularly considered models below.

## Self-coupled scalar
The first particle model that we consider is the simplest: a scalar particle that interacts with itself through a two-to-two coupling. This can be described by the Lagrangian
$$ \mathcal{L}_{\text{int}} = -\frac{\lambda}{4!} \varphi^4. $$
From the Lagrangian, we can read off the Feynman rule for a four-point intersection to have the matrix element $i\mathcal{M} = -i\lambda$, yielding the two-to-two self-interaction differential cross-section
$$ \frac{d\sigma}{d\Omega} = \frac{\lambda^2}{64\pi^2 (4m^2}. $$
Integrating over the solid angle gives a total cross section
$$ \sigma(\varphi\varphi\to\varphi\varphi) = \frac{\lambda^2}{128\pi m^2}. $$
One can easily see that this cross section does not admit any kind of velocity independence. Thus, one could make this model consistent for a small subset of scales (e.g. $\sigma/m \sim 1$ cm$^2$/g for galaxy scales), but then it would necessarily fail on other scales. This makes the model okay only for analyses of limited scales where the cross-section is not expected to vary, but it is generally infeasible as a solution to the small-scale problems.

## Light mediator
Perhaps the simplest model with a theory rich enough to solve all of the observed problems is one wherein dark matter self-interactions are mediated by a light particle. We will consider a model where dark matter is represented by $\chi$ and has mass $m_\chi$, and the mediator field is $\phi$ with mass $m_\phi$. This theory works with both scalar and vector mediators, depending on what specific theory one wants to consider. Perhaps the best motivated origin for such a model is one where the dark matter particle is charged under a spontaneously broken $U(1)$ symmetry and the mediator arises as the corresponding gauge boson (Tulin Yu 2018).

Such a model would have an interaction Lagrangian given by 
$$ \mathcal{L}_{\text{int}} = \cdots $$
where we let the coupling constant be $g_\chi$. In the non-relativistic limit, the interaction is well-approximated by the Yukawa potential (Tulin, Yu, Zurek 2013a,b)
$$ V(r) = \pm \frac{\alpha_\chi}{r} e^{-m_\phi r}, $$
where $\alpha_chi \equiv g_\chi^2/4\pi$ is the dark fine structure constant. The $\pm$ will be set depending on whether the interaction is attractive or repulsive. For a scalar $\phi$, the potential is attractive and the sign is $(-)$. For vector $\phi$, the potential is attractive $(+)$ for $\chi\overline{\chi}$ scattering and repulsive $(-)$ for $\chi\chi$ and $\overline{\chi}\overline{\chi}$ scattering. 

Using the Yukawa potential, we can obtain the Born differential cross section in the limit that $\alpha_\chi m_\chi / m_\phi \ll 1$ to be (Tulin, Yu 2018)
$$ \frac{d\sigma}{d\Omega} = \cdots $$
An important implication of this formula is that the mediator mass must be positive, i.e.$m_\phi > 0$. If instead $m_\phi = 0$, we would then find that $d\sigma/d\Omega$ \propto v_{\text{rel}}^{-4}$, which is far too strong at small velocities to admit a solution which is consistent with observations. A small but nonzero mediator mass $m_\phi$, on the other hand, allows us to “soften” this velocity-dependence to admit a more consistent model.

While quite simple, it has been shown in (todo cite) that it is possible for it to simultaneously accommodate all important observations and solve the small scale problems.

## Strong interactions
Some of the richest theories for self-interacting dark matter candidates that one can consider are non-Abelian gauge theories where the dark matter candidates arise as composite bound states. In these theories, the self-interaction manifests as a strong interaction.

The motivation for considering such a model comes from our experience with QCD and the visible sector (todo cite). For a dark matter model to be a good candidate, it must be stable over the lifetime of the Universe and be neutral under standard model phenomena. Further, we desire models in which the particles exhibit strong self-interactions. These are all properties exhibited by particles in the visible sector under QCD, so it makes sense to consider a similar theory to describe our dark matter candidate. However, we do not necessarily know the gauge group or particle properties of dark matter, leaving us a great freedom to vary the model significantly. Many of the resulting models thus have interesting and unique new physics, though these details are greatly model-dependent. 

The primary free parameters of models of this kind are the confinement scale $\Lambda$ (different from the cosmological constant), and the dark quark mass(es). In the event that our “dark QCD” contains no analogue to electromagnetic/weak interactions, meson-like bound states of the dark quarks could be stable (todo cite). These mesons can be classified as loosely pion-like, where $m \ll \Lambda$, or quarkonium-like, where $m \gg \Lambda$ (todo cite). There are several proposed models for each of these scenarios; one of the more well-known is the strongly-interacting massive particle, or SIMP, where the dark matter candidate is pion-like and many non-Abelian theories are possible. 

Our non-Abelian model may instead look quite similar to visible QCD, wherein the primary stable bound states are baryonic in nature. In (todo cite), it is noted that the advantage of such models is that “dark matter is automatically sufficiently stable, and no further ultraviolet model-building is needed.” One such dark baryon model is “Stealth Dark Matter,” proposed by the LSD collaboration, which is a scalar dark baryon under a confining $SU(4)$ theory. This theory is named _stealth_ dark matter because it is found that the baryons are safe from direct detection, though it does predict a spectrum of lighter meson particles that would be possible to detect at colliders (todo cite).

The third class of candidate particles that has received attention are dark glueballs. Glueballs are bound states of only gluons and are predicted to exist in QCD, but are very difficult to detect. Dark glueballs would then be bound states of dark gluons. Such a model is possible if all the dark fermions in the theory have masses significantly larger than $\Lambda$. In this case, glueballs may become stable under an accidental symmetry like baryons, allowing them to be the primary dark matter candidate.e

The observables that could result from the above considerations are as diverse as the models themselves. One aspect of these models that wee have not considered is what the interactions with the standard model could look like. Some models predict the dark matter candidate to be neutral under standard model interactions, but its constituents to be charged. In such a case, the model would have a coupling to the photon, and it would be possible to directly detect the particle. We may also consider the case where our theory predicts fundamental fermions. It is plausible that these fermions would obtain at least part of their mass through a coupling to the Higgs boson, again providing a mechanism by which we could directly detect the particles. Kribs and Neil provide more details of these observables, as well as collider-specific results, in (todo cite).

# In simulation
todo discuss how SIDM is represented in GIZMO
following section 2 [https://ui.adsabs.harvard.edu/abs/2013MNRAS.430...81R/abstract](https://ui.adsabs.harvard.edu/abs/2013MNRAS.430...81R/abstract)!!

In this work, we will be exploring the introduction of self-interaction in predictions about the infall of the Sagittarius dwarf galaxy and, specifically, the formation of its stream. This is done through the use of N-body simulations. As such, we present a description of how these self-interactions are modeled in simulation. We choose to use GIZMO (todo cite) for our simulations, and the implementation of self-interactions therein is the one described by (Rocha et al. 2012). 

In our simulation, we consider some number of “macro-particles,” each of which represents an ensemble of dark matter particles, a patch of the dark matter phase-space density. We let each macro-particle have mass $m_p$, and we keep this mass consistent across all dark matter macro-particles. Since we consider the macro-particle as representing a patch of the phase-space density, we consider its position to be centered at some point $\mathbf{x}$ but spread out according to a kernel $W(r,h)$. Here, $r$ is the distance from the center of the macro-particle and $h$ is a smoothing length. In GADGET-2, from which GIZMO is built, the kernel is given by 
$$ W(r,h) = \cdots $$
todo find whether this is the same for GIZMO. todo find how $h$ is determined in GIZMO. The velocity of the macro-particle, on the other hand, is taken to be a delta function, such that the macro-particles have a single defined velocity.

When the patches represented by two macro-particles overlap, we can compute the interaction rate between them. The rate of scattering of a macro-particle $j$ off a target particle $i$ is given by
$$ \Gamma(i|j) = (\sigma/m) m_p |\mathbf{v}_i - \mathbf{v}_j| g_{ji}, $$
where $\sigma/m$ is the familiar cross-section to mass ratio and $g_{ij}$ is a number density factor whose purpose is account for the overlap of the two macro-particles’ smoothing kernels. It is given by
$$ g_{ji} = \int_{0}^{h} d^3 \mathbf{x}' \, W(|\mathbf{x}'|, h) \, W(|\delta \mathbf{x}_{ji} + \mathbf{x}'|, h), $$
with $\delta \mathbf{x}_{ji}$ the displacement vector between the macro-particle positions.

Over the course of a time step $\delta t$, the probability of an interaction of macro-particle $j$ off target macro-particle $i$ is given by 
$$ P(i|j) = \Gamma(i|j) \, \delta t. $$
The total probability of interaction between these two particles in this time step, then, would be the average of the two directed probabilities, i.e.
$$ P_{ij} = \tfrac{1}{2} \left( P(i|j) + P(j|i) \right). $$
To actually represent the interaction, then, one draws a random number and adjusts the velocities of the particles if the number lies below the probability. The velocities are adjusted in a manner consistent with an elastic scattering, isotropic in the center of mass frame.

More details are presented in (Rocha et al. 2012), including the derivation of the scattering rate formula from the Boltzmann equation. We use the implementation which is packaged with the publicly-available GIZMO simulation suite.