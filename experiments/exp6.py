
from exps import *
from generate_demographics_graph import *
import numpy
import matplotlib.pyplot as plt
import networkx as nx
import random

params = dict(
    # graph params
    model = "basic",
    n = 100,
    p1 = 0.01 / 2,
    p2 = 0.01 / 2,
    M = 0.2, 
    E = 2, 
    N = 100,
    H = 0.5,
    ALPHA = 1,
    PD = 0.6, 
    ED = 1,
    p_inter = 0.0001,
    # seeding params
    seed1 = 0.3,
    seed2 = 0.3,
    # IC params
    alpha = 0.1,
    reprs = 10000,
    communities = [list(range(0, 100)), list(range(100, 200))]) 


params["graph"] = demographics_communities(params)
params["seeds"] = set_seeds(params)

ic_result = run_experiment(params)
ic_result = f(ic_result, params)
c1, c2, ma, mi = split_result_by_demographics(ic_result, params)


plt.figure()
plot_demographics_dists(ic_result, params)
plt.show()
