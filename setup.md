# Setup

Documenting my set-up process to get things running on Della.

Current progress:
- [x] Python and conda virtual environments
- [ ] Gadget-2 (*need files*)
- [x] DICE
- [ ] GalactICS
    - [ ] GalactICS (*weird compile errors*)
    - [ ] GadgetConverters (*need files*)
- [ ] FlexCombineGalaxies (*doesn't seem to exist??*)
- [x] Gizmo


# Python and Virtual Environment

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


# Gadget-2 Installation

The following will download and extract Gadget-2 on Della.

```
cd ~
wget "https://wwwmpa.mpa-garching.mpg.de/gadget/gadget-2.0.7.tar.gz"
tar -xzf gadget-2.0.7.tar.gz
```

Prof.  Lisanti's installation instructions indicate that she has an edited
Makefile and shell script for using it, but I don't know where to get those. For
now, then, I'm just moving on.


# DICE Installation

[Source](https://bitbucket.org/vperret/dice/wiki/Compile%20&%20Install)

We can install DICE with the following code:

```
cd ~
mkdir local
git clone https://bitbucket.org/vperret/dice
cd dice
mkdir build
cd build
cmake ..
make
make install
cd ~
echo 'export PATH=$HOME/local/bin:$PATH' >> .bashrc # add `dice` to path
```

At this point, `dice` should be installed and executable.


# GalactICS Installation

First, download and unzip the package from the internet with this code.

```
cd ~
wget "https://www.dropbox.com/s/b5coije85zumzq2/GalactICSPackage_Mar2019.zip"
unzip GalactICSPackage_Mar2019.zip
```

### GalactICS

*[Source](https://www.dropbox.com/s/b5coije85zumzq2/GalactICSPackage_Mar2019.zip?dl=0&file_subpath=%2FGalactICSPackage_Mar2019%2FREADMES%2FGalactICSInstallationReadMe.txt)*

Navigate to the GalactICS `src` folder.

```
cd GalactICSPackage_Mar2019/GalactICS/src
```

Run `pwd`, and copy the output to clipboard. Then, open the `makeflags` file 
with `emacs` or `vim` and change the second line to `LocalPath=<paste>`.

Now, while still in the `src` directory, we run

```
make clean
make 
make install
```

This doesn't work for me and I don't know why... `make` returns the following
error message

```
mv dbh ../;
mv: cannot stat ‘dbh’: No such file or directory
make[1]: *** [install] Error 1
```

If I simply re-run it, it still throws an error, and I'm still unable to run
`make install`. Not sure why that's happening. For now, let's move on.

### GadgetConverters

```
cd ~/GalactICSPackage_Mar2019/GadgetConverters/src
mv General/GeneralMath/matrixMath.f General/GeneralMath/MatrixMath.f
```

At this point, Prof. Lisanti's instructions say that I should get a `makeflags`
and a `Makefile` from her to use to compile.


# FlexCombineGalaxies Installation

I am supposed to be able to find this at 
[this link](https://bitbucket.org/surftour/flexcombinegalaxies),
but it appears that that repository no longer exists. Not sure what to do about
that.


# Gizmo Installation

[Gizmo User Guide](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html)

We download Gizmo with 

```
git clone https://bitbucket.org/phopkins/gizmo-public.git
cd gizmo-public
```

This comes with a `Template_Config.sh` file. For now, I simply copied this to
`Config.sh` without modifications. Then, using Prof. Lisanti's `Makefile` and
`Makefile.systype` files, we can compile GIZMO.


