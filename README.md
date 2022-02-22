# ricky

[![Python package](https://github.com/agile-geoscience/ricky/actions/workflows/python-package.yml/badge.svg)](https://github.com/agile-geoscience/ricky/actions/workflows/python-package.yml)
[![Build docs](https://github.com/agile-geoscience/ricky/actions/workflows/sphinx_docs.yml/badge.svg)](https://github.com/agile-geoscience/ricky/actions/workflows/sphinx_docs.yml)

[![PyPI version](https://img.shields.io/pypi/v/ricky.svg)](https://pypi.org/project/ricky//)
[![PyPI versions](https://img.shields.io/pypi/pyversions/ricky.svg)](https://pypi.org/project/ricky//)
[![PyPI license](https://img.shields.io/pypi/l/ricky.svg)](https://pypi.org/project/ricky/)

Popular, and unpopular, wavelets for seismic geophysics. All the wavelets!


## Installation

You can install this package with `pip`:

    pip install ricky

Ricky depends on `xarray`.


## Documentation

Read [the documentation](https://code.agilescientific.com/ricky)


## Example

You can produce a Ricker wavelet with:

    import ricky
    w = ricky.ricker(duration=0.256, dt=0.002, f=25)
    w.plot()


## Testing

You can run the tests (requires `pytest` and `pytest-cov`) with

    python run_tests.py


## Building

This repo uses PEP 518-style packaging. [Read more about this](https://setuptools.pypa.io/en/latest/build_meta.html) and [about Python packaging in general](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

Building the project requires `build`, so first:

    pip install build

Then to build `ricky` locally:

    python -m build

The builds both `.tar.gz` and `.whl` files, either of which you can install with `pip`.


## Continuous integration

This repo has two GitHub 'workflows' or 'actions':

- Push to `main`: Run all tests on all version of Python. This is the **Run tests** workflow.
- Publish a new release: Build and upload to PyPI. This is the **Publish to PyPI** workflow. Publish using the GitHub interface, for example ([read more](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
