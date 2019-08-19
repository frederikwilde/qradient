# qradient
Unlike the name suggests this has become a package to simulate parametrized quantum circuits.

## setup
This package was developed with Python 3.7.2. With older versions you might have to adapt some versions of the dependencies.
Simply edit the version number in the 'requirements.txt' file.

1) Clone the repository.
2) Create virtual environment and activate it.
3) Install the dependencies specified in 'requirements.txt'. With Pip simply run

```pip install -r requirements.txt```.

With Conda you might be able to run

```while read requirement; do conda install --yes $requirement; done < requirements.txt```,

as described [here](https://gist.github.com/luiscape/19d2d73a8c7b59411a2fb73a697f5ed4), though I have not tried this.

## contributing
If you want to contribute, please open a seperate branch and send a pull request.
