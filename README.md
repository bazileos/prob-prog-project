# CS4340 Project Repository - Group 4

The project and the respective problem solution scripts can be run via the Julia singularity image that was provided with the project assignment.

## Singularity image definition file

The singularity image definition can be found in the `cs4340.def` file. It tells the image builder which Julia packages to install after building the image itself. Since the project uses PyCall, after the package installations, it also installs the necessary Python packages via the Conda environment embedded in the PyCall Julia package.

### Potentially missing packages

On some systems it was experienced, that even after running the post-build script for the package installations, the Julia instance running inside the image could not reach these packages. In case this happens, one needs to install the packages manually. For this, enter the Julia REPL in the image by running the following script:

```sh
singularity run cs4340.sif
```

Then the respective packages can be installed via these two commands:

```sh
import Pkg; Pkg.add(["Gen", "StatsBase", "PyCall", "Conda", "Plots", "Distributions"]);
```

```sh
using Conda; Conda.add(["scipy=1.8", "matplotlib", "seaborn", "gym"]);
```

## Running problem solutions

The Julia scripts for the problem solutions can be found in the root folder of the repository, named in accordance with the assignment requirements. These scripts can be run directly with the singularity image. To provide the input data files, the source folder in the host system needs to be binded with the singularity file system. Provided that these input files are placed in the root folder of the project as well, the scripts can be run in the following way:

### Epidemiological Model: SIR

```sh
singularity run --bind [REPOSITORY_PATH]:/mnt cs4340.sif /mnt/sir.jl /mnt/[INPUT_FILE] 500
```

### Symbolic Regression

```sh
singularity run --bind [REPOSITORY_PATH]:/mnt cs4340.sif /mnt/sr.jl /mnt/[INPUT_FILE] 500
```

The X-Y value pairs in the rows of the file are `;` separated.

### Building Safety

```sh
singularity run --bind [REPOSITORY_PATH]:/mnt cs4340.sif /mnt/building.jl /mnt/[INPUT_FILE] 100
```

The observations for the 6 components in the rows of the file are `;` separated.

### Reasoning about Geological Structures

```sh
singularity run --bind [REPOSITORY_PATH]:/mnt cs4340.sif /mnt/geo.jl /mnt/[INPUT_FILENAME] 200
```