#pragma once

#include <vector>
#include <queue>
#include "graph.h"

std::vector<float> run_ic(std::vector<int> &seeds, float alpha, int n_rounds, const Graph &graph);

std::vector<float> run_thresh_sim_ic(
    std::vector<int> &seeds,
    std::vector<float> &thresholds,
    float alpha, int n_rounds,
    const Graph &graph);

template <typename T>
std::vector<float> run_ic_fun(
    std::vector<int> &seeds, float alpha, int n_rounds, const Graph &graph,
    T fun)
{
  int v = graph.n_nodes();
  
  std::vector<float> values(v, 0);

  for (int i = 0; i < n_rounds; ++i) {
    std::vector<bool> is_on(v, false);
    std::queue<int> on_nodes;
    std::vector<int> hits(v, 0);
    
    for (auto &seed: seeds) {
      is_on[seed] = true;
      hits[seed]++;
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

    for (int i = 0; i < graph.m_edges.size(); ++i) {
      auto &out_edges = graph.m_edges[i];
      values[i] += fun(out_edges, hits);
    }
  }

  std::vector<float> result;
  for (auto &value: values) {
    result.push_back(value / (float) n_rounds);
  }
  return result;
}
