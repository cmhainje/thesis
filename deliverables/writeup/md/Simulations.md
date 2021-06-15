# Full infall simulations
# Description and initial results

With the equilibrated and merged initial conditions for both cuspy (CDM) and cored (SIDM) galaxies, we now carry out our full simulations of the Sgr infall. We will consider _three_ mergers: the cuspy initial conditions evolved using CDM microphysics, the cored initial conditions evolved with CDM microphysics, and the cored initial conditions evolved with SIDM microphysics. As before, we take $\sigma / m$ = 10 cm$^2$/g in the SIDM case. These three mergers will be referred to as CDM/cuspy, CDM/cored, and SIDM respectively. By performing all three simulations, we will ideally be able to identify whether certain discrepancies between the CDM/cuspy and the SIDM runs are the result of a cored initial profile or from the inclusion of self-interactions.

For each merger, the infall is simulated for 10 Gyr, with snapshots saved every 0.978 Gyr. In Figure todo, we show the positions of the stars of Sgr in the orbital plane at several times for each merger. Similarly, we show the positions of the Sgr dark matter particles in Figure todo.

todo Figure: sgr stars for all three mergers at t = 0, 3, 6, 9

todo Figure: same but with DM

Even from these plots, some differences appear to emerge. 

todo talk more about this

# Identifying the Sgr progenitor

A key part of analyzing these data is to understand the trajectory and evolution of the Sgr progenitor in particular. As such, we desire a method for successfully identifying the position of the Sgr progenitor throughout its evolution. This is less straightforward than it may sound because of the strong effects of tidal stripping. These mean that we need to identify which particles are stripped or bound to the progenitor at any given point and omit those particles which have been stripped from our calculation of the progenitor position. In our tests, we tried a few different methods which we will describe here.

The first method that we tried was simply stripping at a fixed radius. The algorithm works as follows.



how to identify the progenitor?
tried a few ways
* stripping at King radius
* stripping at decreasing radius
* stripping at fixed radius
* identifying com by dm instead of stars
show plots of fractional mass over time, etc.
results
* trajectory of progenitor in orbital plane
* distance from mw over time
discuss important differences, point out sidm growing apocenter

# Comparison to stream data

comparison to streamfinder data