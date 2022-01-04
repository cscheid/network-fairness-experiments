ic: main main_fun

main: main.cc graph.cc independent_cascade.cc
	g++ -std=c++17 -g -O2 $^ -o ic

main_fun: main_fun.cc graph.cc independent_cascade.cc
	g++ -std=c++17 -g -O2 $^ -o ic_fun
