FROM python:latest

RUN pip install -e ./src/copper_100 ./src/copper_111 ./src/nickel_111 ./src/surface_potential_analysis
