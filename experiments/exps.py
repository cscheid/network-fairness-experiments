from utils import *
import numpy
import glob

import random
from operator import itemgetter

import matplotlib.pyplot as plt
import seaborn as sns

def collect_neighbor_data(result, params):
    graph = read_graph(params["graph"])
    return list(list(result[i] for i in out_nodes) for out_nodes in graph)

def hist(v, **kwargs):
    defaults = dict(
        histtype = "stepfilled", 
        alpha = 0.3,
        bins = 20, 
        range = (0, 1), 
        density = True
    )
    for (k, value) in kwargs.items():
        defaults[k] = value
    
    plt.hist(v, **defaults)

def split_result_by_communities(experiment_result, params):
    seeds = params["seeds"]
    communities = params["communities"]
    result_list = []
    for community in communities:
        community_result = []
        for node in community:
            if node in seeds:
                continue
            v = experiment_result[node]
            if v == None:
                continue
            if v == -1:
                # this comes from the c++ computation
                continue
            community_result.append(v)
        result_list.append(community_result)
    return result_list
    
def plot_community_dists(result, params):
    for result in split_result_by_communities(result, params):
        hist(result)

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

def ic_fun(graph_file_name, seeds, alpha, reps, fun):
    return run_cmd(["../ic/ic_fun", graph_file_name, seeds, str(alpha), str(reps), fun])

def read_array(filename):
    with open(filename) as f:
        return list(float(v) for v in f.readline().split())
        
def set_seeds(params):
    seed1 = params["seed1"]
    seed2 = params["seed2"]
    n1 = params.get('n1', params.get("n", 0))
    n2 = params.get('n2', params.get("n", 0))
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
    n1 = params.get('n1', params.get("n", 0))
    n2 = params.get('n2', params.get("n", 0))
    return community_graph(params["p_inter"], n1, params["p1"], n2, params["p2"])

def read_graph(name):
    g = []
    with open(name, "r") as f:
        l = f.readline()
        if '\t' in l:
            l = l.split('\t')
            n = int(l[0])
            d = int(l[1])
        else:
            n = int(l)
            d = int(f.readline())
        for i in range(n):
            g.append([])
        for l in f:
            f, t = list(int(v) for v in l.strip().split())
            g[f].append(t)
            if d == 0:
                g[t].append(f)
    return g
                

##############################################################################
# collateral consquence functions

def id(r, params):
    return r
def square(r, params):
    return numpy.array(r) ** 2
def f_mean(r, params):
    r = collect_neighbor_data(r, params)
    def mean_or_none(v):
        if len(v) == 0:
            return None
        return numpy.mean(v)
    return list(mean_or_none(v) for v in r)
def f_min(r, params):
    r = collect_neighbor_data(r, params)
    def min_or_none(v):
        if len(v) == 0:
            return None
        return numpy.min(v)
    return list(min_or_none(v) for v in r)

##############################################################################
# graph IO

def graph_to_edge_list(network):
    result = []
    for (node_id, neighbors) in enumerate(network):
        for neighbor in neighbors:
            result.append([node_id, neighbor])
    return result
    
def write_output(G, filename):
    with open(filename, 'w') as txt_file:
        num_of_nodes = len(G.nodes)
        directed = 0
        txt_file.write("{}\t{}\n".format(num_of_nodes, directed))
        for edge in G.edges:
            txt_file.write("{}\t{}\n".format(edge[0], edge[1]))

def write_graph(network, filename):
    with open(filename, 'w') as txt_file:
        num_of_nodes = len(network)
        directed = 0
        txt_file.write("{}\t{}\n".format(num_of_nodes, directed))
        for f, l in enumerate(network):
            for t in l:
                txt_file.write("{}\t{}\n".format(f, t))
    
# def network_into_file(network):
#     n = temp_name(".txt")
#     g = nx.Graph(graph_to_edge_list(network))
#     write_output(g, n)
#     return n

##############################################################################
# Data access

graphs = []
model = ["SBM", "LFR"]
communities = ["Isolated_communities", "More_connected_communities"]

for m in model:
    for c in communities:
        files = glob.glob(f'../data/reference_communities/{c}/{m}/Run_*/twocommunities_edgelist.txt')
        l = []
        graphs.append(
            dict(model=m,
                 community_type=c,
                 files=files))
