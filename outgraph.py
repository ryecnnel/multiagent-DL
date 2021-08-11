# cording: utf-8

import os,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import networkx as nx


"""
start:FLAGS
"""

flag_cmpltGraph = 1
flag_blncdGraph = 1
flag_unblncdGraph = 1
flag_single = 1
flag_online = 1

"""
end:FLAGS
"""


path = os.path.dirname(os.path.abspath(__file__)) + '/data'

(n) = np.loadtxt(path+'/n.dat').astype(int)

iter_per_epoch = np.loadtxt(path+'/iterperepoch.dat')
(max_epochs) = np.loadtxt(path+'/max_epochs.dat').astype(int)

suffix = "_withCom"
if flag_cmpltGraph:
    prefix = "Complete"
    (maxdeg_cm) = np.loadtxt(path+'/'+prefix+'Graph_maxdeg.dat')
    Gadj_cm = np.loadtxt(path+'/'+prefix+'Graph_adjMat.dat')
    suffix2 = "_complete"
    train_acc_comp = np.loadtxt(path+'/train_acc_list'+suffix+suffix2+'.dat')
    test_acc_comp = np.loadtxt(path+'/test_acc_list'+suffix+suffix2+'.dat')
    train_loss_comp = np.loadtxt(path+'/train_loss_list'+suffix+suffix2+'.dat')
    loss_each_func_comp = np.loadtxt(path+'/loss_each_func_list'+suffix+suffix2+'.dat')
if flag_blncdGraph:
    prefix = "Balanced"
    (maxdeg_bl) = np.loadtxt(path+'/'+prefix+'Graph_maxdeg.dat')
    Gadj_bl = np.loadtxt(path+'/'+prefix+'Graph_adjMat.dat')
    suffix2 = "_balanced"
    train_acc_bld = np.loadtxt(path+'/train_acc_list'+suffix+suffix2+'.dat')
    test_acc_bld = np.loadtxt(path+'/test_acc_list'+suffix+suffix2+'.dat')
    train_loss_bld = np.loadtxt(path+'/train_loss_list'+suffix+suffix2+'.dat')
    loss_each_func_bld = np.loadtxt(path+'/loss_each_func_list'+suffix+suffix2+'.dat')
if flag_unblncdGraph:
    prefix = ""
    (maxdeg) = np.loadtxt(path+'/Graph_maxdeg.dat')
    Gadj = np.loadtxt(path+'/Graph_adjMat.dat')
    suffix2 = "_unbalanced"
    train_acc_ubl = np.loadtxt(path+'/train_acc_list'+suffix+suffix2+'.dat')
    test_acc_ubl = np.loadtxt(path+'/test_acc_list'+suffix+suffix2+'.dat')
    train_loss_ubl = np.loadtxt(path+'/train_loss_list'+suffix+suffix2+'.dat')
    loss_each_func_ubl = np.loadtxt(path+'/loss_each_func_list'+suffix+suffix2+'.dat')

suffix = "_withoutCom"
if flag_single:
    suffix2 = ""
    train_acc_sin = np.loadtxt(path+'/train_acc_list'+suffix+suffix2+'.dat')
    test_acc_sin = np.loadtxt(path+'/test_acc_list'+suffix+suffix2+'.dat')
    train_loss_sin = np.loadtxt(path+'/train_loss_list'+suffix+suffix2+'.dat')
    loss_each_func_sin = np.loadtxt(path+'/loss_each_func_list'+suffix+suffix2+'.dat')


#####

### Do not use type-3-fonts.
# plt.rcParams['ps.useafm'] = True
# plt.rcParams['pdf.use14corefonts'] = True
# plt.rcParams['text.usetex'] = True

font = {# 'family' : 'serif',
        # 'weight' : 'bold',
        'size'   : 15
        } 

plt.rc('font', **font)

ls = ['-', '--', '-.', ':', '-', '--', '-.', ':', ]
colors=[["darkred","indianred"],["darkgreen","mediumseagreen"],["midnightblue","slateblue"]]

#####


if flag_cmpltGraph:
    plt.figure(figsize=(6,6))
    G = nx.from_numpy_matrix(Gadj_cm, create_using=nx.Graph)
    pos = nx.circular_layout(G)
    nx.draw(G, pos, arrowsize=30, node_size=500)
    plt.savefig(path+"/figs/graph_n"+str(n)+"_complete.png")
    plt.savefig(path+"/figs/graph_n"+str(n)+"_complete.pdf")
    plt.show(block=False)
if flag_blncdGraph:
    plt.figure(figsize=(6,6))
    G = nx.from_numpy_matrix(Gadj_bl, create_using=nx.DiGraph)
    pos = nx.circular_layout(G)
    nx.draw(G, pos, arrowsize=30, node_size=500)
    plt.savefig(path+"/figs/graph_n"+str(n)+"_balanced.png")
    plt.savefig(path+"/figs/graph_n"+str(n)+"_balanced.pdf")
    plt.show(block=False)
if flag_unblncdGraph:
    plt.figure(figsize=(6,6))
    G = nx.from_numpy_matrix(Gadj, create_using=nx.DiGraph)
    pos = nx.circular_layout(G)
    nx.draw(G, pos, arrowsize=30, node_size=500)
    plt.savefig(path+"/figs/graph_n"+str(n)+"_unbalanced.png")
    plt.savefig(path+"/figs/graph_n"+str(n)+"_unbalanced.pdf")
    plt.show(block=False)
    
# input()
# sys.exit()

###

markers = {'train': 'o', 'test': 's'}
mevery = 500 if flag_online else 10

plt.figure(figsize=(8,6))
x = np.arange(max_epochs+1)
if flag_cmpltGraph:
    plt.plot(x, train_acc_comp, marker='o', color='indianred', label='complete_train', markevery=mevery, linestyle=ls[0], linewidth=1.0)
    plt.plot(x, test_acc_comp, marker='s', color='indianred', label='complete_test', markevery=mevery, linestyle=ls[1], linewidth=1.0)
if flag_blncdGraph:
    plt.plot(x, train_acc_bld, marker='o', color=colors[0][0], label='balanced_train', markevery=mevery, linestyle=ls[0], linewidth=1.0)
    plt.plot(x, test_acc_bld, marker='s', color=colors[0][0], label='balanced_test', markevery=mevery, linestyle=ls[1], linewidth=1.0)
if flag_unblncdGraph:
    plt.plot(x, train_acc_ubl, marker='o', color=colors[1][0], label='unbalanced_train', markevery=mevery, linestyle=ls[0], linewidth=1.0)
    plt.plot(x, test_acc_ubl, marker='s', color=colors[1][0], label='unbalanced_test', markevery=mevery, linestyle=ls[1], linewidth=1.0)
if flag_single:
    plt.plot(x, train_acc_sin, marker='o', color=colors[2][0], label='single_train', markevery=mevery, linestyle=ls[0])
    plt.plot(x, test_acc_sin, marker='s', color=colors[2][0], label='single_test', markevery=mevery, linestyle=ls[1], linewidth=1.0)
if not flag_online: plt.xlabel("epochs k")
else:   plt.xlabel("iterations k")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
### x軸に100刻みにで小目盛り(minor locator)表示
# plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(50))
### y軸に1刻みにで小目盛り(minor locator)表示
plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(0.1))
### 小目盛りに対してグリッド表示
plt.grid(which='minor', axis='y')
plt.legend(loc='lower right')
plt.savefig(path+"/figs/accuracy.png")
plt.savefig(path+"/figs/accuracy.pdf")
plt.show(block=False)

###

plt.figure(figsize=(8,6))
if flag_cmpltGraph:
    plt.plot(x, train_acc_comp, marker='o', color='indianred', label='complete', markevery=mevery, linestyle=ls[0], linewidth=1.0)
if flag_blncdGraph:
    plt.plot(x, train_acc_bld, marker='o', color=colors[0][0], label='balanced', markevery=mevery, linestyle=ls[1], linewidth=1.0)
if flag_unblncdGraph:
    plt.plot(x, train_acc_ubl, marker='o', color=colors[1][0], label='unbalanced', markevery=mevery, linestyle=ls[2], linewidth=1.0)
if flag_single:
    plt.plot(x, train_acc_sin, marker='o', color=colors[2][0], label='single', markevery=mevery, linestyle=ls[3], linewidth=1.0)
if not flag_online: plt.xlabel("epochs k")
else:   plt.xlabel("iterations k")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
### x軸に100刻みにで小目盛り(minor locator)表示
# plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(50))
### y軸に1刻みにで小目盛り(minor locator)表示
plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(0.1))
### 小目盛りに対してグリッド表示
plt.grid(which='minor', axis='y')
plt.legend(loc='lower right')
plt.savefig(path+"/figs/accuracy_train.png")
plt.savefig(path+"/figs/accuracy_train.pdf")
plt.show(block=False)

###

plt.figure(figsize=(8,6))
if flag_cmpltGraph:
    plt.plot(x, test_acc_comp, marker='s', color='indianred', label='complete', markevery=mevery, linestyle=ls[0], linewidth=1.0)
if flag_blncdGraph:
    plt.plot(x, test_acc_bld, marker='s', color=colors[0][0], label='balanced', markevery=mevery, linestyle=ls[1], linewidth=1.0)
if flag_unblncdGraph:
    plt.plot(x, test_acc_ubl, marker='s', color=colors[1][0], label='unbalanced', markevery=mevery, linestyle=ls[2], linewidth=1.0)
if flag_single:
    plt.plot(x, test_acc_sin, marker='s', color=colors[2][0], label='single', markevery=mevery, linestyle=ls[3], linewidth=1.0)
if not flag_online: plt.xlabel("epochs k")
else:   plt.xlabel("iterations k")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
### x軸に100刻みにで小目盛り(minor locator)表示
# plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(50))
### y軸に1刻みにで小目盛り(minor locator)表示
plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(0.1))
### 小目盛りに対してグリッド表示
plt.grid(which='minor', axis='y')
plt.legend(loc='lower right')
plt.savefig(path+"/figs/accuracy_test.png")
plt.savefig(path+"/figs/accuracy_test.pdf")
plt.show(block=False)

###

plt.figure(figsize=(8,6))
if not flag_online:
    x = np.arange(iter_per_epoch * (max_epochs-1) +1)
    if flag_cmpltGraph:
        plt.plot(x, train_loss_comp, color=colors[0][1], label='complete', linewidth=1.0, linestyle=ls[0], alpha = 1.0, marker='s', markevery=mevery)
    if flag_blncdGraph:
        plt.plot(x, train_loss_bld, color=colors[0][0], label='balanced', linewidth=1.0, linestyle=ls[1], alpha = 1.0, marker='s', markevery=mevery)
    if flag_unblncdGraph:
        plt.plot(x, train_loss_ubl, color=colors[1][0], label='unbalanced', linewidth=1.0, linestyle=ls[2], alpha = 1.0, marker='s', markevery=mevery)    
    if flag_single:
        plt.plot(x, train_loss_sin, color=colors[2][0], label='single', linewidth=1.0, linestyle=ls[3], alpha = 1.0, marker='s', markevery=mevery)
else:
    x = np.arange(max_epochs)
    if flag_cmpltGraph:
        plt.plot(x, loss_each_func_comp, color=colors[0][1], label='complete', linewidth=1.0, linestyle=ls[0], alpha = 1.0, marker='s', markevery=mevery)
    if flag_blncdGraph:
        plt.plot(x, loss_each_func_bld, color=colors[0][0], label='balanced', linewidth=1.0, linestyle=ls[1], alpha = 1.0, marker='s', markevery=mevery)
    if flag_unblncdGraph:
        plt.plot(x, loss_each_func_ubl, color=colors[1][0], label='unbalanced', linewidth=1.0, linestyle=ls[2], alpha = 1.0, marker='s', markevery=mevery)    
    if flag_single:
        plt.plot(x, loss_each_func_sin, color=colors[2][0], label='single', linewidth=1.0, linestyle=ls[3], alpha = 1.0, marker='s', markevery=mevery)
plt.xlabel("iterations k")
plt.ylabel("loss (mean squared error)")
### y軸に1刻みにで小目盛り(minor locator)表示
plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(200))
### 小目盛りに対してグリッド表示
plt.grid(which='minor', axis='y')
plt.ylim(0, 1.0)
plt.legend(loc='upper right')
plt.savefig(path+"/figs/loss.png")
plt.savefig(path+"/figs/loss.pdf")
plt.show(block=False)

###

plt.figure(figsize=(8,6))
if not flag_online:
    x = np.arange(iter_per_epoch * (max_epochs-1) +1)
else:
    x = np.arange(max_epochs)
if flag_single:
    p1, = plt.plot(x, loss_each_func_sin, color=colors[2][0], label='single', linewidth=1.0, linestyle=ls[3], alpha = 1.0)
if flag_blncdGraph:
    p2, = plt.plot(x, loss_each_func_bld, color=colors[0][0], label='balanced', linewidth=1.0, linestyle=ls[1], alpha = 1.0)
if flag_cmpltGraph:
    p3, = plt.plot(x, loss_each_func_comp, color=colors[0][1], label='complete', linewidth=1.0, linestyle=ls[0], alpha = 1.0)
if flag_unblncdGraph:
    p4, = plt.plot(x, loss_each_func_ubl, color=colors[1][0], label='unbalanced', linewidth=1.0, linestyle=ls[2], alpha = 1.0)    
plt.xlabel("iterations k")
plt.ylabel("loss (mean squared error)")
### y軸に1刻みにで小目盛り(minor locator)表示
plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(0.1))
### 小目盛りに対してグリッド表示
plt.grid(which='minor', axis='y')
# plt.ylim(0, 0.7)
plt.legend([p3,p2,p4,p1], ["complete", "balanced", "unbalanced", "single"], loc='upper right')
plt.savefig(path+"/figs/loss_each_step.png")
plt.savefig(path+"/figs/loss_each_step.pdf")
plt.show()
