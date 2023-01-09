# surface_potential_analysis

Code to produce localized wavepackets for periodic potential energy surfaces.

## Installing

To install the two packages run

```shell
pip install -e ./src/surface_potential_analysis
pip install ./src/hamiltonian_generator
```

Note hamiltonian_generator is a dependency of surface_potential_analysis but I couldn't figure out how to add local deps to pyproject.toml. Best practice is to do this all in a virtual environment.

Functions can then be imported and run from main.py.
