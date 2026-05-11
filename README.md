# thermohaline-steps

A python toolbox to detect thermohaline staircases and calculate some statistical properties of them.

### Install

To install the latest release from PyPI

```bash
pip install thremohalinesteps
```

For an editable local install

```bash
pip install -e .
```

An example notebook demonstrating functionality can be found in the `examples` directory

### Aims
- [x] Replicate classification system of van der Boog et al.
- [x] Generalise to accept generic evenly sampled CTD profiles as input
- [x] Accept irregularly sampled input
- [x] Test using synthetic profiles
- [x] Distribute via PyPI

### Related work

This toolbox is used in the paper "Glider observations of thermohaline staircases in the tropical North Atlantic using an automated classifier" Callum Rollo, Karen J. Heywood & Rob A. Hall 2022 https://doi.org/10.5194/gi-11-359-2022

It is a genrealised reimplemenation of the original work by Carine van Der Boog et al. 2020

Original paper by van Der Boog et al. https://doi.org/10.5194/essd-13-43-2021

Original codebase by van Der Boog https://doi.org/10.5281/zenodo.4286170


### License

Licensed under GPL v3

