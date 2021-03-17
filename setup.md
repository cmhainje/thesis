# Setup

Documenting my set-up process to get things running on Della.

Contents:
- Python and conda virtual environments
- GalactICS
    - GalactICS
    - GadgetConverters
- Gizmo

Note that there are `Makefile` and `makeflags` files in the `setup-files/`
directory of this repository.


## Python and Virtual Environment

[Source](https://researchcomputing.princeton.edu/python)

Upon logging in, we need to set-up Anaconda and create a new virtual
environment. Anaconda is pre-installed on Della, so we use

```
module load anaconda3
conda create --name <venv_name> python=3.7
conda activate <venv_name>
```

In the future, we will need to do the following to get the virtual environment
running: 

```
module load anaconda3
conda activate <venv_name>
``` 

when logging in. When in the virtual environment, you can install Python 
packages as normal using `pip install <package>`.

```
pip install numpy pandas scipy h5py matplotlib tqdm jupyter
``` 

On top of the standard `pip`-able packages, there are two packages for GIZMO
analysis that we will need to install, and these were not so easy for me to
install. Here is how I got it to work.

```
cd ~
mkdir pypackages
cd pypackages
git clone https://bitbucket.org/awetzel/gizmo_analysis.git
git clone https://bitbucket.org/awetzel/utilities.git
cp gizmo_analysis/setup.py .
python setup.py develop
cp utilities/setup.py .
python setup.py develop
rm setup.py
```


## GalactICS Installation

First, download and unzip the package from the internet with this code.

```
cd ~
wget "https://www.dropbox.com/s/b5coije85zumzq2/GalactICSPackage_Mar2019.zip"
unzip GalactICSPackage_Mar2019.zip
```

### GalactICS

[Source](https://www.dropbox.com/s/b5coije85zumzq2/GalactICSPackage_Mar2019.zip?dl=0&file_subpath=%2FGalactICSPackage_Mar2019%2FREADMES%2FGalactICSInstallationReadMe.txt)

Copy the GalactICS `Makefile` and `makeflags` into
`GalactICSPackage_Mar2019/GalactICS/src` (replacing the ones that are already
there). Then, open the `makeflags` file with `vim` or `emacs` and edit
`LocalPath` to match your local path. From inside the `src` directory, run

```
make clean
make 
make install
```

The `make` command may throw an error the first time; just run it again and it
should compile.


### GadgetConverters

```
cd ~/GalactICSPackage_Mar2019/GadgetConverters/src
mv General/GeneralMath/matrixMath.f General/GeneralMath/MatrixMath.f
```

At this point, copy the GadgetConverters `makeflags` and `Makefile` files into
`GalactICSPackage_Mar2019/GadgetConverters/src`. Be sure to edit the `makeflags`
file as above. Next, we need to fix the Makefile in the `Bin` directory. `cd ../Bin`,
then open `Makefile`. Add `-lm` after in the `ascii2gadget_gasIni` rule, so it 
should look like

```
ascii2gadget_gasIni: ascii2gadget_gasIni.o
	$(CC) $(CFLAGS) -o ascii2gadget_gasIni ascii2gadget_gasIni.o -lm
```

Then, run `make clean`, `make`, `make` and `make install`. (You have to run 
`make` twice, because it always throws an error the first time for some reason, 
I don't know why.) 

## Gizmo Installation

[Gizmo User Guide](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html)

We download Gizmo with 

```
git clone https://bitbucket.org/phopkins/gizmo-public.git
cd gizmo-public
```

This comes with a `Template_Config.sh` file. For now, I simply copied this to
`Config.sh` without modifications. Then, we copy the GIZMO `Makefile` and
`Makefile.systype` files into the source directory, and compile in the usual way
(`make clean`, `make`).

