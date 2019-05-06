========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor|
        |
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/2019sp-final-project-elsalmi/badge/?style=flat
    :target: https://readthedocs.org/projects/2019sp-final-project-elsalmi
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/csci-e-29/2019sp-final-project-elsalmi.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/csci-e-29/2019sp-final-project-elsalmi

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/csci-e-29/2019sp-final-project-elsalmi?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/csci-e-29/2019sp-final-project-elsalmi

.. |version| image:: https://img.shields.io/pypi/v/final-project.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/final-project

.. |commits-since| image:: https://img.shields.io/github/commits-since/csci-e-29/2019sp-final-project-elsalmi/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/csci-e-29/2019sp-final-project-elsalmi/compare/v0.0.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/final-project.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/final-project

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/final-project.svg
    :alt: Supported versions
    :target: https://pypi.org/project/final-project

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/final-project.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/final-project


.. end-badges

An example package. Generated with cookiecutter-pylibrary.

* Free software: BSD 2-Clause License

Installation
============

::

    pip install final-project

Documentation
=============


https://2019sp-final-project-elsalmi.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
