#include "independent_cascade.h"
#include <iostream>
#include <cstdlib>
#include <queue>

using namespace std;

vector<float> run_ic(vector<int> &seeds, float alpha, int n_rounds, const Graph &graph)
{
  int v = graph.n_nodes();
  
  vector<int> hits(v, 0);

  for (int i = 0; i < n_rounds; ++i) {
    vector<bool> is_on(v, false);
    queue<int> on_nodes;
    
    for (auto &seed: seeds) {
      hits[seed]++;
      is_on[seed] = true;
      on_nodes.push(seed);
    }

    while (!on_nodes.empty()) {
      int this_node = on_nodes.front();
      on_nodes.pop();
      for (auto &out_edge: graph.m_edges[this_node]) {
        if (is_on[out_edge])
          continue;
        if (rand() / (float) RAND_MAX <= alpha) {
          is_on[out_edge] = true;
          hits[out_edge]++;
          on_nodes.push(out_edge);
        }
      }
    }
  }

  vector<float> result;
  for (auto &hit: hits) {
    result.push_back(hit / (float) n_rounds);
  }
  return result;
}
