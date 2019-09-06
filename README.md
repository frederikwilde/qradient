# qradient
This package provides a platform for fast simulation of parametrized quantum circuits.
Also, fast optimization is facilitated by efficient gradient calculations, something that cannot be done on a physical quantum device.

## setup for _importing_ the package
1) Clone the repository.
2) Create virtual environment in the repository you want to work in and activate it. E.g. with 'venv': 
```
python3 -m venv my-env
source ./my-env/bin/activate
```
3) Install the qradient package. E.g. with pip
```
pip install -e /path/to/qradient
```

## setup for _working on_ the package
This package was developed with Python 3.7.2. With older versions you might have to adapt some versions of the dependencies.
Simply edit the version number in the 'requirements.txt' file.

1) Clone the repository.
2) Create virtual environment in the package repository ```qradient/``` and activate it. E.g. with 'venv': 
```
python3 -m venv my-env
source ./my-env/bin/activate
```
3) Install the requirements. E.g. with pip:
```
pip install -r requirements.txt
```


## contributing
If you want to contribute, please open a seperate branch and send a pull request.

## a note on jupyter notebooks
The workflow of jupyter notebooks is somewhat unfitting for working with a VCS.
The major problem is, that large outputs, like images, are effectively binary and therefore make merging difficult.
However, notebooks are nice to use for tutorials.

The **policy concerning notebooks** for this repository is that **ALL CELL OUTPUTS SHOULD BE CLEARED** before committing.
This can be done manually or with a simple filter for git, which you can find the instructions for [here](https://intoli.com/blog/jupyter-notebooks-git/).

This also implies that notebooks should not contain code that runs longer than a few seconds.
If you want to put results of expensive computations into a notebook, save it to a file using the [data_storage](https://github.com/frederikwilde/qradient/blob/master/tutorials/data-storage.ipynb) module and then import it in the notebook.
