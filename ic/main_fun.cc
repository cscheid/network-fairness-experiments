#include "graph.h"
#include "independent_cascade.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <string>

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

float min_or_negone(const vector<int> &m_edges,
                  const vector<int> &hits)
{
  if (m_edges.size() == 0) {
    return -1;
  }
  int r = 10000000;
  for (auto &n: m_edges) {
    r = min(hits[n], r);
  }
  return r;
}

float mean_or_negone(const vector<int> &m_edges,
                  const vector<int> &hits)
{
  if (m_edges.size() == 0) {
    return -1;
  }
  int r = 0;
  for (auto &n: m_edges) {
    r += hits[n];
  }
  return float(r) / float(m_edges.size());
}

int main(int argc, char **argv)
{
  if (argc < 6) {
    cout << "Expected at least 5 args" << endl;
    cout << "Usage: " << argv[0] << " graph_in seeds_in alpha nreps fun" << endl;
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

  vector<float> result;

  if (string(argv[5]) == string("min")) {
    result = run_ic_fun(seeds, alpha, nreps, g, &min_or_negone);
  } else if (string(argv[5]) == string("mean")) {
    result = run_ic_fun(seeds, alpha, nreps, g, &mean_or_negone);
  } else {
    cout << "Bad fun: don't know how to run " << argv[5] << endl;
    return 1;
  }
  
  for (auto &v: result) {
    cout << v << " ";
  }
  cout << endl;
  
  return 0;
}
