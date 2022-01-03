#!/usr/bin/env python3

# generate simple community graphs, erdos-renyi within with some per-community p, erdos-renyi without with some overall p
#
# usage:
#

##
# ./$0 inter_community_prob community_size_1 intra_community_prob_1 community_size_2 intra_community_prob_2 ... network
#
##

# outputs graph to stdout

import sys
import random

##
v = sys.argv[1:-1]
network = sys.argv[-1]
##

inter_community_prob = float(v[0])
sizes = list(int(i) for i in v[1::2])
probs = list(float(i) for i in v[2::2])

# G = nx.Graph()
n = sum(sizes)

print(n) # n edges
print(0) # no, undirected

with open(network) as edgelist:
    next(edgelist)
    for line in edgelist:
        if line[-1] == "\n":
            line = line[1:-1]
        line = line.split("\t")
        i, j = int(line[0]), int(line[1])
        print(i, j)
