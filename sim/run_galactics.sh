# remove existing things
cd ~/GalactICSPackage_Mar2019/GalactICS/models/run
make veryclean
rm -rf in.*

# copy over input files
cp -r $1/galactics/in.* .
touch b.dat

# make them
make potential
make disk
make halo

# copy results back
cp disk halo dbh.dat $1/galactics
