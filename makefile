compile: hrnn.cpp
	c++ -O3 -Wall -shared -std=c++11 -fPIC -march=native `python3 -m pybind11 --includes` hrnn.cpp -o hrnn`python3-config --extension-suffix`
installPyblind:
	pip install pybind11
