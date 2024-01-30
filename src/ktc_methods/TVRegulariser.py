"""
I constructed the TV matrix according to https://pubmed.ncbi.nlm.nih.gov/20051330/

adenker@uni-bremen.de
"""

import networkx as nx
import numpy as np 

def create_tv_matrix(mesh):
    g = mesh.g #node coordinates
    H = mesh.H #indices of nodes making up the triangular elements
    
    G = nx.Graph()
    
    for i in range(len(g)):
        G.add_node(i, pos=g[i,:])
    
    for i in range(len(H)):
        G.add_edge(H[i,0], H[i,1], weight= np.linalg.norm(G.nodes[H[i,0]]['pos'] - G.nodes[H[i,1]]['pos']))
        G.add_edge(H[i,1], H[i,2], weight= np.linalg.norm(G.nodes[H[i,1]]['pos'] - G.nodes[H[i,2]]['pos']))
        G.add_edge(H[i,2], H[i,0], weight= np.linalg.norm(G.nodes[H[i,2]]['pos'] - G.nodes[H[i,0]]['pos']))
        
    Ltv = np.zeros((len(G.edges), len(g)))

    for i, edge in enumerate(G.edges):

        Ltv[i, edge[0]] = G.edges[edge]['weight']
        Ltv[i, edge[1]] = -G.edges[edge]['weight']
        
    return Ltv, G
