#include "graph.h"

using namespace std;

Graph read_graph(istream &input)
{
  int v;
  input >> v;
  
  int directed;
  input >> directed;
  
  Graph result(v);
  int from, to;
  
  while (input >> from >> to) {
    result.add_edge(from, to);
    if (!directed) {
      result.add_edge(to, from);
    }
  }

  return result;
}
