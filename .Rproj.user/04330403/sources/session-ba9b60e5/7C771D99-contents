#include "graph.h"
#include "independent_cascade.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;

vector<int> read_seeds(ifstream &in)
{
  vector<int> result;
  int v;
  while (in >> v) {
    result.push_back(v);
  }
  return result;
}

int main(int argc, char **argv)
{
  if (argc < 5) {
    cout << "Expected at least 4 args" << endl;
    cout << "Usage: " << argv[0] << " graph_in seeds_in alpha nreps" << endl;
    return 1;
  }

  /* initialize random seed: */
  srand(time(NULL));
  
  ifstream graph_in(argv[1]);
  ifstream seeds_in(argv[2]);
  Graph g = read_graph(graph_in);
  vector<int> seeds = read_seeds(seeds_in);
  float alpha = stof(argv[3]);
  int nreps = atoi(argv[4]);

  vector<float> result = run_ic(seeds, alpha, nreps, g);
  for (auto &v: result) {
    cout << v << " ";
  }
  cout << endl;
  
  return 0;
}
