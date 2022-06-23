compile: hrnn.cpp
	c++ -O3 -Wall -shared -std=c++11 -fPIC -march=native `python3 -m pybind11 --includes` hrnn.cpp -o hrnn`python3-config --extension-suffix` -lblas
env:
	. ~/intel/oneapi/setvars.sh
mkl:
	icx -O3 -v -Wall --no-undefined -shared -std=c++11 -fPIC -march=native -lpython -L /home/gosti/intel/oneapi/intelpython/latest/lib/python3.9/site-packages/pybind11/include `python3 -m pybind11 --includes` hrnn.cpp -o hrnn`python3-config --extension-suffix`
installPyblind:
	pip install pybind11
