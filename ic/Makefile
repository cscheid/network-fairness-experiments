all: ic ic_fun threshold_sim

ic: main.cc graph.cc independent_cascade.cc
	g++ -std=c++17 -Wall -g -O2 $^ -o ic

ic_fun: main_fun.cc graph.cc independent_cascade.cc
	g++ -std=c++17 -Wall -g -O2 $^ -o ic_fun

threshold_sim: threshold_sim.cc graph.cc independent_cascade.cc
	g++ -std=c++17 -Wall -g -O2 $^ -o threshold_sim
