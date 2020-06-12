# qradient [kɹeɪdi.ənt]
This package provides a platform for fast simulation of parametrized quantum circuits.
Also, fast optimization is facilitated by efficient gradient calculations, something that cannot be done on a physical quantum device.

## status of development
This package is under development and many components are far from optimal. Future versions will be code-breaking.

## installing the package (recommendation)
1) Clone this repository.
2) Create a virtual environment in the repository you want to work in and activate it. E.g. with 'venv':
```
python3 -m venv my-env
source ./my-env/bin/activate
```
3) Install the package into your environment
```
pip install -e /path/to/qradient
```
where ```-e``` installs it in 'editible' mode, i.e. you can make changes to the package without having to reinstall it afterwards.

## contributing
If you want to contribute, please create a new branch and make a pull request.

## runtimes
There are certainly quantum-circuit simulators out there that are much faster than this one.
However, when your circuit is parametrized by many parameters, this package might provide the fastest way to access the gradient.
To show this consider the ```RandomRotations``` (inspired by a recent [paper](https://www.nature.com/articles/s41467-018-07090-4)) circuit sketched below, where each R is an arbitrary Pauli rotation, each parametrized by one individual parameter.
Finally a Pauli ZZ measurement is performed on the first and second qubit (the choice of the observable only marginally affects the runtime).
The circuit can be built by the following piece of code:
```python
import numpy as np
from qradient.circuit_logic import RandomRotations

qubit_num, layer_num = 3, 3
zz_interactions = np.full((qubit_num, qubit_num), None)  # empty adjacency matrix
zz_interactions[0, 1] = 1.  # only couple the first to the second qubit
axes = np.random.randint(0, 3, size=(layer_num, qubit_num))  # pick rotation axes
circuit = RandomRotations({'zz': interactions}, axes)
```
The method whose runtimes are shown below is `circuit.gradient()`, which returns the expectation value and the full gradient of the circuit with respect to all `qubit_num * layer_num` parameters.

| circuit | runtimes |
| ------- | -------- |
| ![RandomRotations circuit]() | ![runtime diagram]() |

*These numbers were obtained on a 2,3 GHz Quad-Core Intel Core i5 processor*

## a note on jupyter notebooks
The workflow of jupyter notebooks is somewhat unfitting for working with a VCS.
The major problem is, that large outputs, like images, are effectively binary and therefore make merging difficult.
However, notebooks are nice to use for tutorials.

The **policy concerning notebooks** for this repository is that **ALL CELL OUTPUTS SHOULD BE CLEARED** before committing.
This can be done manually or with a simple filter for git, which you can find the instructions for [here](https://intoli.com/blog/jupyter-notebooks-git/).

This also implies that notebooks should not contain code that runs longer than a few seconds.
If you want to put results of expensive computations into a notebook, save it to a file using a file export package (e.g. [h5py](https://docs.h5py.org/en/stable/index.html) for HDF5 import/export) and then import it in the notebook.

## contact
If you have questions or comments, feel free to open a GitHub issue or send me a mail at: wilde.physics[ät]gmail.com

## acknowledgements and citing
I, Frederik Wilde, acknowledge funding from the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany´s Excellence Strategy – MATH+ : The Berlin Mathematics Research Center, EXC-2046/1 – project ID: 390685689.

If you use this code for work that you publish, please cite this repository.
```
@misc{qradient,
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/frederikwilde/qradient}}
}
```

## license
This work is subject to the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.txt) as specified in the [LICENSE](https://github.com/frederikwilde/qradient/blob/master/LICENSE) file.
