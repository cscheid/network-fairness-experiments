#pragma once

#include <vector>
#include "graph.h"

std::vector<float> run_ic(std::vector<int> &seeds, float alpha, int n_rounds, const Graph &graph);
