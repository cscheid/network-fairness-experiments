# code for network models from Wang et. al. 2022
# creates a network containing two mutually exclusive groups: 'majority' and 'minority'

# M: proportion of 'minority' nodes
# E: number of edges for each new node
# N: number of nodes in network
# H: homophily - H = 0.5 for random mixing, 1.0 for perfectly homophilic.
# ALPHA: preferential attachment strength 
# PD: diversification probability 
# ED: diversified links - ED = 0 is Homophily BA

from sqlite3 import paramstyle
from exps import *
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import copy

params = dict(
    # graph params
    model = "basic",
    n = 100,
    p1 = 0.01 / 2,
    p2 = 0.01 / 2,
    M = 0.5, 
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

models = ["basic","homophily","diversified"]

def randomNetwork(m, e0, e1, N):
    e_dict = {0: e0, 1: e1}
    G = nx.Graph()
    G.add_nodes_from([(0, {'group': 0}), (1, {'group': 1})])
    G.add_edge(0, 1)
    for t in range(N):
        # choose which group this new node is in
        if random.random() < m:
            g = 1  # 1 is minority group
        else:
            g = 0
        group = nx.get_node_attributes(G, "group")
        # random select nodes
        selected_nodes = random.choices(sorted(group), k=e_dict[g])
        # add edges
        G.add_node(2 + t, group=g)
        G.add_edges_from([(2 + t, target) for target in selected_nodes])
    return G

def homophilyBA(m, e0, e1, h0, h1, alpha, N):
    homo_dict = {0: h0, 1: h1}
    e_dict = {0: e0, 1: e1}
    G = nx.Graph()
    G.add_nodes_from([(0, {'group': 0}), (1, {'group': 1})])
    G.add_edge(0, 1)
    for t in range(N):
        # choose which group this new node is in
        if random.random() < m:
            g = 1  # 1 is minority group
        else:
            g = 0
        # calculate connection probability to each node
        degree = dict(G.degree())
        group = nx.get_node_attributes(G, "group")
        info = []
        for node in sorted(group):
            g_n = group[node]
            d_n = degree[node]
            h = homo_dict[g]
            if g_n == g:
                p = h * d_n**alpha
            else:
                p = (1 - h) * d_n**alpha
            info.append(p)
        # select node based on the probability
        selected_nodes = random.choices(
            sorted(group), weights=info, k=e_dict[g])  # currently doing select with replacement
        # add edges
        G.add_node(2 + t, group=g)
        G.add_edges_from([(2 + t, target) for target in selected_nodes])
    return G

def DiversifiedHomophilyBA(m, e0, e1, h0, h1, alpha, po, eo0, eo1, N, weighted=True):
    homo_dict = {0: h0, 1: h1}
    e_dict = {0: e0, 1: e1}
    eo_dict = {0: eo0, 1: eo1}
    portion = {0: po, 1: 1 - po}  # different group 1-po
    G = nx.Graph()
    G.add_nodes_from([(0, {'group': 0}), (1, {'group': 1})])
    G.add_edge(0, 1)
    for t in range(N):
        # choose which group this new node is in
        if random.random() < m:
            g = 1  # 1 is minority group
        else:
            g = 0
        # calculate connection probability to each node
        degree = dict(G.degree())
        group = nx.get_node_attributes(G, "group")
        info = []
        for node in sorted(group):
            g_n = group[node]
            d_n = degree[node]
            h = homo_dict[g]
            if g_n == g:
                p = h * d_n**alpha
            else:
                p = (1 - h) * d_n**alpha
            info.append(p)
        # select node based on the probability
        selected_nodes = random.choices(
            sorted(group), weights=info, k=e_dict[g] - eo_dict[g])
        # select friends of friends who are of opposite group of me based on
        # degree
        info_dict = {}
        for n in selected_nodes:
            degree_n = degree[n]
            neighbors = list(G.neighbors(n))
            for neighbor in neighbors:
                # if group[neighbor] != g:
                degree_neighbor = degree[neighbor]
                distance = abs(degree_neighbor - degree_n)
                if neighbor in info_dict:
                    if distance > info_dict[neighbor]:
                        info_dict[neighbor] = portion[
                            group[neighbor] == g] / (distance + 0.1)
                else:
                    info_dict[neighbor] = portion[
                        group[neighbor] == g] / (distance + 0.1)
        if len(info_dict) > 0:
            fof, info = list(zip(*info_dict.items()))
            fof, info = list(fof), list(info)
            if weighted:
                selected_fof = random.choices(
                    fof, weights=info, k=min(len(info), eo_dict[g]))
            else:
                selected_fof = random.choices(
                    fof, k=min(len(info), eo_dict[g]))
        else:
            selected_fof = []
        # add edges
        G.add_node(2 + t, group=g)
        G.add_edges_from([(2 + t, target)
                          for target in selected_nodes + selected_fof])
    return G

def demographics_communities(params):
    #create two communities each containing a 'majority' and 'minority' population  
    if params["model"]=="diversified":
        C1 = DiversifiedHomophilyBA(params["M"], params["E"], params["E"], 0.5, 0.5, params["ALPHA"], params["PD"], params["ED"], params["ED"], params["N"]-2)
        C2 = DiversifiedHomophilyBA(params["M"], params["E"], params["E"], 0.5, 0.5, params["ALPHA"], params["PD"], params["ED"], params["ED"], params["N"]-2)
    if params["model"]=="homophily":
        C1 = homophilyBA(params["M"], params["E"], params["E"], params["H"], params["H"], params["ALPHA"], params["N"]-2)
        C2 = homophilyBA(params["M"], params["E"], params["E"], params["H"], params["H"], params["ALPHA"], params["N"]-2)
    else:
        C1 = randomNetwork(params["M"], params["E"], params["E"], params["N"]-2)
        C2 = randomNetwork(params["M"], params["E"], params["E"], params["N"]-2)
    #combine these two communities into a single network
    rename = {}
    N = params["N"]
    for i in range(N+2): 
        rename[i] = (N+2+i)
        C2 = nx.relabel_nodes(C2,rename)
        C = nx.compose(C1,C2) #preserves attributes
        for i in range(N+2):
            for j in range(N,N+N+2):
                p = random.random()
                if p < params["p_inter"]:
                    C.add_edge(i,j)
    #taking only largest connected component
    #C = [C.subgraph(c).copy() for c in nx.connected_components(C)]
    with open("../data/demographics/dem_edgelist.txt", 'w') as file:
        file.write("{}\t{}\n".format(C.number_of_nodes(), 0))
        for edge in C.edges:
            file.write("{}\t{}\n".format(edge[0], edge[1]))    
    with open("../data/demographics/dem_groups.txt",'w') as file:
        for node,data in C.nodes(data="group"):
            file.write("{}\t{}\n".format(node,data))  
    print(params["n"]) # n edges
    print(0) # no, undirected 
    [print(edge[0],edge[1]) for edge in C.edges] 

def run_experiment_batch(conf):
    result = {}
    for (k, params) in conf.items():
        params = copy.copy(params)
        params["ic_result"] = run_experiment(params)
        result[k] = params
    return result

def run_thresh_experiment_batch(conf):
    result = {}
    for (k, params) in conf.items():
        params = copy.copy(params)
        degrees = numpy.array(list(len(n) for n in read_graph(params["graph"])))
        params["thresholds"] = numpy.random.random(len(degrees)) * degrees
        params["ic_result"] = run_thresh_experiment(params)
        result[k] = params
    return result

def figure_1(params, xlabel):
    c1, c2, ma, mi = split_result_by_demographics(params["ic_result"], params)
    hist(c1)
    hist(c2)
    hist(ma)
    hist(mi)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")

def figure_2(conf, ylabel):
    x_values_1 = []
    x_values_2 = []
    y_values_1 = []
    y_values_2 = []
    for (k, v) in conf.items():
        c1, c2, ma, mi = split_result_by_demographics(v["ic_result"], v)
        c1 = numpy.mean(c1)
        c2 = numpy.mean(c2)
        #ma = numpy.mean(ma)
        #mi = numpy.mean(mi)
        x_values_1.append(v["factor"])
        x_values_2.append(v["factor"])
        y_values_1.append(c1)
        y_values_2.append(c2)
    sns.lineplot(x = x_values_1, y = y_values_1, estimator = numpy.mean, ci = 95)
    sns.lineplot(x = x_values_2, y = y_values_2, estimator = numpy.mean, ci = 95)
    sns.scatterplot(x = x_values_1, y = y_values_1, hue="group")
    sns.scatterplot(x = x_values_2, y = y_values_2, hue="group")
    plt.xlabel("Factor")
    plt.ylabel(ylabel)

def figure_3(conf):
    x_values = []
    y_values = []
    for (k, v) in conf.items():
        c1, c2 = split_result_by_communities(v["ic_result"], v)
        c1 = numpy.mean(c1)
        c2 = numpy.mean(c2)
        f = min(c1, c2) / max(c1, c2)
        x_values.append(v["factor"])
        y_values.append(f)
    sns.lineplot(x = x_values, y = y_values, estimator = numpy.mean, ci = 95)
    sns.scatterplot(x = x_values, y = y_values)
    plt.ylim(0,1)
    plt.xlabel("Factor")
    plt.ylabel("Fairness")

def run_and_plot(conf, k, prefix):
    r = run_experiment_batch(conf)
    plt.figure()
    figure_1(r[k], "Access probability")
    plt.savefig(prefix + "-fig-1.png")
    plt.show()
    
    plt.figure()
    figure_2(r, "Access")
    plt.savefig(prefix + "-fig-2.png")
    plt.show()

    plt.figure()
    figure_3(r)
    plt.savefig(prefix + "-fig-3.png")
    plt.show()

#Seeding experiment
def configure_experiment_seeding(factors, graphs):
    result = {}
    params = dict(
        n1 = 1000,
        n2 = 1000)
    for f in factors:
        for g in graphs:
            # n = read_graph(g)
            params["seed1"] = f * 0.1
            params["seed2"] = 0.1 
            params["graph"] = g
            params["seeds"] = set_seeds(params)
            params["factor"] = f
            params["graph"] = g
            params["alpha"] = 0.1
            params["reprs"] = n_reps
            params["communities"] = [list(range(0, 1000)), list(range(1000, 2000))]
            result[(f, g)] = copy.copy(params)
    return result

#Node removal experiment
def delete_nodes_from_network(network):
    network = read_graph(network)
    edge_list = graph_to_edge_list(network)
    nodes_to_delete = set() # solve_this_later()
    edge_list = list(
        edge for edge in edge_list
        if (edge[0] not in nodes_to_delete and
            edge[1] not in nodes_to_delete))
    g = nx.Graph(edge_list)
    write_output(g, "output.txt")
    return "output.txt"

factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def run_experiment_range(network, communities, factors, f):
    c1s = []
    c2s = []
    for factor in factors:
        #params = make_params(factor)
        #params["n"] = 1000
        #params["n1"] = 1000
        #params["n2"] = 1000
        params["seed1"] = factor
        params["seed2"] = 1-factor
        params["graph"] = delete_nodes_from_network(network)
        params["communities"] = communities
        params["seeds"] = set_seeds(params)
        ic_result = run_experiment(params)
        ic_result = f(ic_result, params)
        c1, c2, ma, mi = split_result_by_communities(ic_result, params)
        c1s.append(numpy.mean(c1))
        c2s.append(numpy.mean(c2))
        ma.append(numpy.mean(ma))
        mi.append(numpy.mean(mi))
    return c1s, c2s, ma, mi

def plot_curve(network, community, f, label):
    c1s, c2s, ma, mi = run_experiment_range(network, community, factors, f)
    plt.figure()
    plt.plot(factors, c1s)
    plt.plot(factors, c2s)
    plt.plot(factors, ma)
    plt.plot(factors, mi)
    plt.xlabel("Shrinkage factor")
    plt.ylabel(label)
    plt.show()

#Create graph
params["graph"] = demographics_communities(params)
#Set seeds
params["seeds"] = set_seeds(params)
#Run experiments
#ic_result = run_experiment(params)
#ic_result = f(ic_result, params)
#c1, c2, ma, mi = split_result_by_demographics(ic_result, params)
#print(itemgetter('graph', 'seeds', 'alpha', 'reprs')(params))

run_experiment_batch(params)
#run_thresh_experiment_batch(params)
