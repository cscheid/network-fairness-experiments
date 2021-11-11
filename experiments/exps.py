from utils import *

import random
from operator import itemgetter

import matplotlib.pyplot as plt

def collect_neighbor_data(result, params):
    graph = read_graph(params["graph"])
    seeds = set
    return list(list(result[i] for i in out_nodes) for out_nodes in graph)

def hist(v):
    kwargs = dict(
        histtype = "stepfilled", 
        alpha = 0.3,
        bins = 20, 
        range = (0, 1), 
        density=True)
    plt.hist(v, **kwargs)

def split_result_by_communities(result, params):
    seeds = set(params["seeds"])
    n = params.get("n1", params.get("n", None))
    c1 = list(x for (i, x) in enumerate(result[:n]) if i not in seeds and x is not None)
    c2 = list(x for (i, x) in enumerate(result[n:]) if (n + i) not in seeds and x is not None)
    return c1, c2
    
def plot_community_dists(result, params):
    c1, c2 = split_result_by_communities(result, params)
    hist(c1)
    hist(c2)
    
def array_into_file(vec):
    n = temp_name()
    with open(n, "w") as f:
        for v in vec:
            f.write(str(v))
            f.write(" ")
    return n

def community_graph(*args):
    return run_cmd(["../generate_community_graph.py"] + list(str(i) for i in args))

def ic(graph_file_name, seeds, alpha, reps):
    return run_cmd(["../ic/ic", graph_file_name, seeds, str(alpha), str(reps)])

def read_array(filename):
    with open(filename) as f:
        return list(float(v) for v in f.readline().split())
        
def set_seeds(params):
    seed1, seed2 = itemgetter('seed1', 'seed2')(params)
    n1 = params.get('n1', params["n"])
    n2 = params.get('n2', params["n"])
    seeds = []
    for i in range(n1):
        # seed community1 with probability seed1
        if random.random() < seed1:
            seeds.append(i)
    for i in range(n2):
        # seed community2 with probability seed2
        if random.random() < seed2:
            seeds.append(n1 + i)
    return sorted(seeds)

def run_experiment(params):
    graph, seeds, alpha, reprs = itemgetter('graph', 'seeds', 'alpha', 'reprs')(params)
    return read_array(ic(graph, array_into_file(seeds), alpha, reprs))

def two_communities(params):
    n1 = params.get('n1', params["n"])
    n2 = params.get('n2', params["n"])
    return community_graph(params["p_inter"], n1, params["p1"], n2, params["p2"])

def read_graph(name):
    g = []
    with open(name, "r") as f:
        n = int(f.readline())
        d = int(f.readline())
        for i in range(n):
            g.append([])
        for l in f:
            f, t = list(int(v) for v in l.strip().split())
            g[f].append(t)
            if d == 0:
                g[t].append(f)
    return g
                
