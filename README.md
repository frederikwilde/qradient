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

## a note on jupyter notebooks
The workflow of jupyter notebooks is somewhat unfitting for working with a VCS.
The major problem is, that large outputs, like images, are effectively binary and therefore make merging difficult.
However, notebooks are nice to use for tutorials.

The **policy concerning notebooks** for this repository is that **ALL CELL OUTPUTS SHOULD BE CLEARED** before committing.
This can be done manually or with a simple filter for git, which you can find the instructions for [here](https://intoli.com/blog/jupyter-notebooks-git/).

This also implies that notebooks should not contain code that runs longer than a few seconds.
If you want to put results of expensive computations into a notebook, save it to a file using the [data_storage](https://github.com/frederikwilde/qradient/blob/master/tutorials/data-storage.ipynb) module and then import it in the notebook.

## acknowledgements and citing
I acknowledge funding from the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany´s Excellence Strategy – MATH+ : The Berlin Mathematics Research Center, EXC-2046/1 – project ID: 390685689.

If you use this code for work that you publish, please cite this repository.
```
@misc{qradient,
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/frederikwilde/qradient}}
}
```
