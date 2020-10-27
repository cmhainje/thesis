currentdir=$(pwd)

# remove existing things
cd ~/GalactICSPackage_Mar2019/GalactICS/models/run
make veryclean
rm -rf in.*

# copy over input files
cp -r $currentdir/galactics/in.* .

# make them
make potential
make halo

# copy results back
cp halo $currentdir/galactics
cp dbh.dat $currentdir/galactics
