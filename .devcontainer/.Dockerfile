FROM python:latest

RUN pip install -e ./src/copper_100 ./src/copper_111 ./src/platinum_111 ./src/ruthenium_111 ./src/nickel_111

RUN pip install -e ./src/surface_potential_analysis
