ic: main main_fun threshold_sim

main: main.cc graph.cc independent_cascade.cc
	g++ -std=c++17 -g -O2 $^ -o ic

main_fun: main_fun.cc graph.cc independent_cascade.cc
	g++ -std=c++17 -g -O2 $^ -o ic_fun

threshold_sim: threshold_sim.cc graph.cc independent_cascade.cc
	g++ -std=c++17 -g -O2 $^ -o threshold_sim
