#include "graph.h"
#include "independent_cascade.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <string>

using namespace std;

template <typename T>
vector<T> read_vector(ifstream &in)
{
  vector<T> result;
  T v;
  while (in >> v) {
    result.push_back(v);
  }
  return result;
}

int main(int argc, char **argv)
{
  if (argc < 6) {
    cout << "Expected at least 5 args" << endl;
    cout << "Usage: " << argv[0] << " graph_in seeds_in threshs alpha nreps" << endl;
    return 1;
  }

  /* initialize random seed: */
  srand(time(NULL));
  
  ifstream graph_in(argv[1]);
  ifstream seeds_in(argv[2]);
  ifstream threshs_in(argv[3]);
  Graph g = read_graph(graph_in);
  vector<int> seeds = read_vector<int>(seeds_in);
  vector<float> thresholds = read_vector<float>(threshs_in);  
  float alpha = stof(argv[4]);
  int nreps = atoi(argv[5]);

  vector<float> result;

  result = run_thresh_sim_ic(seeds, thresholds, alpha, nreps, g);
  
  for (auto &v: result) {
    cout << v << " ";
  }
  cout << endl;
  
  return 0;
}
