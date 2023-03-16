#!/usr/bin/env python3

# generate simple community graphs, erdos-renyi within with some per-community p, erdos-renyi without with some overall p
#
# usage:
#
# ./$0 inter_community_prob community_size_1 intra_community_prob_1 community_size_2 intra_community_prob_2 ...
#
# outputs graph to stdout

import sys
import random

v = sys.argv[1:]

inter_community_prob = float(v[0])
sizes = list(int(i) for i in v[1::2])
probs = list(float(i) for i in v[2::2])

# G = nx.Graph()
n = sum(sizes)

def range_over_communities(sizes):
    index = 0
    for (community, sz) in enumerate(sizes):
        for _ in range(sz):
            yield (index, community)
            index += 1

print(n) # n edges
print(0) # no, undirected

# print edges    
for i, c1 in range_over_communities(sizes):
    for j, c2 in range_over_communities(sizes):
        p = random.random()
        if c1 == c2:
            if p < probs[c1]:
                print(i, j)
        else:
            if p < inter_community_prob:
                print(i, j)
