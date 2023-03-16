#pragma once

#include <vector>
#include <iostream>
#include <istream>

struct Graph
{
  std::vector<std::vector<int> > m_edges;

  size_t n_nodes() const { return m_edges.size(); }

  explicit Graph(int n)
      : m_edges(n)
  {}

  void add_edge(int from, int to)
  {
    m_edges[from].push_back(to);
  }
};

Graph read_graph(std::istream &input);
