# Artificial-Life-and-Foundation-Models

A group project for COMP0258: Open-Endedness and General Intelligence, a course at UCL led by Professor Tim Rocktäschel.

"Open-endedness (OE) is the ability of a system to keep generating interesting artifacts forever, which is one of the defining features of natural evolution [Stanley et al., 2017]."

- Resources we used for inspiration and reviewed: https://docs.google.com/document/d/1GNeawJdXXLfGvWVEroBYWnJFeIx6geIDxgJdd3e7O1c/edit?tab=t.0

- Assignment Instructions: https://docs.google.com/document/d/1KWJ4eVg3fOkxuVOcFGFFyC3k4vTbnthGplZYqAAAIlg/edit?tab=t.0

## How To Run Different Parts of Code

In order to run jupyter notebooks it is as simple as doing the following commands:
```sh
poetry install
poetry run jupyter notebook
```

Although if you wanted to make this a global setting and use this vitrual environment for everything I would do the following:
```sh
poetry run ipython kernel install --user --name=<KERNEL_NAME>
jupyter notebook
```
This will make the default python for running your notebooks on the jupyter platform the virtual environment's python.

### Running Python Scripts

To run different python scripts one would do:
```sh
poetry run python3 SCRIPT.py
```