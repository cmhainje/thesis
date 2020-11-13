currentdir=$(pwd)

# remove existing things
cd ~/GalactICSPackage_Mar2019/GalactICS/models/run
make veryclean
rm -rf in.*

# copy over input files
cp -r $currentdir/galactics/in.* .
<<<<<<< HEAD
touch b.dat
=======
>>>>>>> 7307f7c4702a96cbeb178b29e96673b9f162f825

# make them
make potential
make disk
make halo

# copy results back
<<<<<<< HEAD
cp disk halo dbh.dat $currentdir/galactics
=======
cp halo $currentdir/galactics
cp dbh.dat $currentdir/galactics
>>>>>>> 7307f7c4702a96cbeb178b29e96673b9f162f825
