#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import network.networkgraph as NetworkGraph

path = os.path.dirname(os.path.abspath(__file__))


def main():
    n = 10
    makeDir()
    makeNetwork(n,mode="ub")
    makeNetwork(n,mode="b")
    makeNetwork(n,mode="c")
    dpath = path+'/data'
    np.savetxt(dpath+'/n.dat', [n])


def makeDir():
    if not os.path.isdir(path+'/data'):
        os.mkdir(path+'/data')

    dpath = path+'/data'
    if not os.path.isdir(dpath+'/figs'):
        os.mkdir(dpath+'/figs')


def makeNetwork(n, mode="ub"):
    if mode == "ub" or mode == "unbalanced":
        (G, adjMat, maxdeg) = NetworkGraph.connected_directed_networkgraph(n,k=3,p=0.5)
        suffix = ""
        prefix = ""
    elif mode == "b" or mode == "balanced":
        (G, adjMat, maxdeg) = NetworkGraph.connected_balanced_networkgraph(n,k=3)
        suffix = "_balanced"
        prefix = "Balanced"
    elif mode == "ws" or mode == "watts-strogatz":
        (G, adjMat, maxdeg) = NetworkGraph.connected_wattsstrogatz_networkgraph(n,k=4,p=0.4)
        suffix = "_ws"
        prefix = "Ws"
    elif mode == "c" or mode == "complete":
        (G, adjMat, maxdeg) = NetworkGraph.complete_networkgraph(n,isDirected=False)
        suffix = "_complete"
        prefix = "Complete"
    else:
        raise ValueError("Unknown mode:str in makeNetwork.")
    
    dpath = path+'/data'
    
    plt.figure(figsize=(6,6))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, font_size=8)
    plt.savefig(dpath+"/figs/graph_n"+str(n)+suffix+".png")
    plt.savefig(dpath+"/figs/graph_n"+str(n)+suffix+".pdf")
    plt.show(block=False)

    np.savetxt(dpath+'/'+prefix+'Graph_adjMat.dat', adjMat)
    np.savetxt(dpath+'/'+prefix+'Graph_maxdeg.dat', [maxdeg])


if __name__ == "__main__":
    main()
    input("::Press ENTER to exit::")