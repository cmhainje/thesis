# Setup

Documenting my set-up process to get things running on Della.

Contents:
- Python and conda virtual environments
- GalactICS
    - GalactICS
    - GadgetConverters
- Gizmo


## Python and Virtual Environment

[Source](https://researchcomputing.princeton.edu/python)

Upon logging in, we need to set-up Anaconda and create a new virtual
environment. Anaconda is pre-installed on Della, so we use

```
module load anaconda3
conda create --name <venv_name>
conda activate <venv_name>
```

In the future, we will need to `module load anaconda3` and `conda activate
analysis` when logging in. This should be fine for now; I'll add the packages I
need to install when I install them.


## GalactICS Installation

First, download and unzip the package from the internet with this code.

```
cd ~
wget "https://www.dropbox.com/s/b5coije85zumzq2/GalactICSPackage_Mar2019.zip"
unzip GalactICSPackage_Mar2019.zip
```

### GalactICS

*[Source](https://www.dropbox.com/s/b5coije85zumzq2/GalactICSPackage_Mar2019.zip?dl=0&file_subpath=%2FGalactICSPackage_Mar2019%2FREADMES%2FGalactICSInstallationReadMe.txt)*

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
file like we did above. Then, run `make clean`, `make`, and `make install`. 

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
(`make clean`, `make`, `make install`).

