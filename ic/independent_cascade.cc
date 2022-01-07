#include "independent_cascade.h"
#include <iostream>
#include <cstdlib>

using namespace std;

void run_ic_round(
    vector<int> &seeds, float alpha,
    vector<int> &hits,
    const Graph &graph)
{
  int v = graph.n_nodes();
  
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

std::vector<float> run_thresh_sim_ic(
    std::vector<int> &seeds,
    std::vector<float> &thresholds,
    float alpha, int n_rounds,
    const Graph &graph)
{
  int v = graph.n_nodes();
  
  vector<int> hits(v, 0);
  for (int i = 0; i < n_rounds; ++i) {
    vector<int> inner_hits(v, 0);
    run_ic_round(seeds, alpha, hits, graph);
    for (int j = 0; j < graph.n_nodes(); ++j) {
      int count = 0;
      for (auto &out: graph.m_edges[j]) {
        count += inner_hits[out];
      }
      if (count > thresholds[j]) {
        hits[j]++;
      }
    }
  }

  vector<float> result;
  for (auto &hit: hits) {
    result.push_back(hit / (float) n_rounds);
  }
  return result;
}

    

vector<float> run_ic(vector<int> &seeds, float alpha, int n_rounds, const Graph &graph)
{
  int v = graph.n_nodes();
  
  vector<int> hits(v, 0);

  for (int i = 0; i < n_rounds; ++i) {
    run_ic_round(seeds, alpha, hits, graph);
  }

  vector<float> result;
  for (auto &hit: hits) {
    result.push_back(hit / (float) n_rounds);
  }
  return result;
}

