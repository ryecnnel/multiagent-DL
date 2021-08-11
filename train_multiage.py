# cording: utf-8
# pylint: disable=E1103
# (eliminate no-member error)

import os,sys
import numpy as np
import matplotlib.pyplot as plt
import network.networkgraph
from network.agent import Agent
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

from argparse import ArgumentParser
import time

# from progressbar import ProgressBar

path = os.path.dirname(os.path.abspath(__file__))


"""
start:FLAGS
"""

# type_Graph = "complete"
type_Graph = "balanced"
# type_Graph = "unbalanced"
# type_Graph = "watts-strogatz"
flag_staticStep = 0
flag_communicate = 1
flag_overfittest = 0
flag_online = 1

"""
end:FLAGS
"""

def main(flag_bycmdline = False, type_Graph=type_Graph,
         flag_staticStep = flag_staticStep,flag_communicate = flag_communicate,
         flag_overfittest = flag_overfittest,flag_online = flag_online):

    dpath = path+'/data'

    ### read graph
    if type_Graph in ('c', 'complete'):
        (n) = np.loadtxt(dpath+'/n.dat').astype(int)
        (maxdeg) = np.loadtxt(dpath+'/CompleteGraph_maxdeg.dat')
        Gadj = np.loadtxt(dpath+'/CompleteGraph_adjMat.dat')
    elif type_Graph in ('b', 'balanced'):
        (n) = np.loadtxt(dpath+'/n.dat').astype(int)
        (maxdeg) = np.loadtxt(dpath+'/BalancedGraph_maxdeg.dat')
        Gadj = np.loadtxt(dpath+'/BalancedGraph_adjMat.dat')
    elif type_Graph in ('ub', 'unbalanced'):
        (n) = np.loadtxt(dpath+'/n.dat').astype(int)
        (maxdeg) = np.loadtxt(dpath+'/Graph_maxdeg.dat')
        Gadj = np.loadtxt(dpath+'/Graph_adjMat.dat')
    elif type_Graph in ('ws', 'watts-strogatz'):
        (n) = np.loadtxt(dpath+'/n.dat').astype(int)
        (maxdeg) = np.loadtxt(dpath+'/WsGraph_maxdeg.dat')
        Gadj = np.loadtxt(dpath+'/WsGraph_adjMat.dat')
    else:
        raise Exception("[Error]: Cannot identify the communication graph type.")

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    if flag_overfittest:
        ### overfit test
        datasize= min(t_train.size, 2500*n)
        x_train = x_train[:datasize]
        t_train = t_train[:datasize]
        ### split data
        x_train_split = np.split(x_train, n)
        t_train_split = np.split(t_train, n)
    else:
        ### split data
        x_train_split = np.split(x_train, n)
        t_train_split = np.split(t_train, n)

    max_epochs = 6000 if flag_online else 151
    # max_epochs = 201
    each_train_size = x_train_split[0].shape[0]
    batch_size = 1 if flag_online else min(100, each_train_size)
    np.savetxt(dpath+'/max_epochs.dat', [max_epochs])

    Agent.n = n
    Agent.maxdeg, Agent.AdjG_init = maxdeg, Gadj
    Agent.train_size, Agent.batch_size = each_train_size, batch_size

    # weight decay（荷重減衰）の設定 =====================
    # 要は正則化パラメータ ===============================
    weight_decay_lambda = 0 # weight decayを使用しない場合
    # weight_decay_lambda = 0.01
    # ====================================================

    if flag_staticStep:
        optimizer = SGD(lr=lambda s:0.1)
    else:
        # optimizer = SGD(lr=lambda s:0.001*1000/(s+1000))
        optimizer = SGD(lr=lambda s:0.005*1000/(s+1000))

    ### TODO:seed も引数で指定できるようにする
    # seed=520
    seed=854
    np.random.seed(seed)
    agents = [Agent(idx, x_train_split[idx], t_train_split[idx], x_test, t_test, 
                    optimizer, weight_decay_lambda, flag_online) for idx in range(n)]

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    loss_each_list = []

    iter_per_epoch = 1 if flag_online else max(each_train_size / batch_size, 1)
    epoch_cnt = 0

    # p = ProgressBar(0,max_epochs-1)
    print("[START]: graph:"+type_Graph+", communicate:"+str(flag_communicate)+", random_seed:"+str(seed)+", static_step:"+str(flag_staticStep)+", online:"+str(flag_online))

    ### cost at k=0
    storeLossAndAccuracy(agents,n,train_acc_list,test_acc_list,train_loss_list,test_loss_list)

    for k in range(1000000000):
        start = time.time()
        for i in range(n):
            agents[i].update(epoch_cnt)

        loss_each_age = tuple(agents[i].layer.varloss for i in range(n))
        loss_each = np.mean(loss_each_age)
        loss_each_list.append(loss_each)

        if k % iter_per_epoch == 0:
            
            if flag_communicate:
                ## communication
                for i in range(n):
                    for j in np.nonzero(agents[i].AdjG)[0]:
                        agents[i].receive(j, *agents[j].send(k,i))
                for i in range(n):
                    agents[i].consensus()

            train_acc, test_acc, train_loss, _ = \
                storeLossAndAccuracy(agents,n,train_acc_list,test_acc_list,train_loss_list,test_loss_list)

            elapsed_time = time.time() - start

            if not flag_online:
                print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))
            else:
                print(str(k) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) 
                             + ", train loss:" + str(train_loss) + ", time:" + str(elapsed_time))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
            # p.update(epoch_cnt)


    # p.finish()
    print("[END]: graph:"+type_Graph+", communicate:"+str(flag_communicate)+", random_seed:"+str(seed)+", static_step:"+str(flag_staticStep)+", online:"+str(flag_online))

    if type_Graph in ('c', 'complete'):
        suffix2 = "_complete"
    elif type_Graph in ('b', 'balanced'):
        suffix2 = "_balanced"
    elif type_Graph in ('ub', 'unbalanced'):
        suffix2 = "_unbalanced"
        
    if flag_communicate:
        suffix = "_withCom"
    else:
        suffix = "_withoutCom"
        suffix2 = ""

    np.savetxt(dpath+'/train_acc_list'+suffix+suffix2+'.dat', train_acc_list)
    np.savetxt(dpath+'/test_acc_list'+suffix+suffix2+'.dat', test_acc_list)
    np.savetxt(dpath+'/train_loss_list'+suffix+suffix2+'.dat', train_loss_list)
    np.savetxt(dpath+'/test_loss_list'+suffix+suffix2+'.dat', test_loss_list)
    np.savetxt(dpath+'/loss_each_func_list'+suffix+suffix2+'.dat', loss_each_list)
    np.savetxt(dpath+'/iterperepoch.dat', [iter_per_epoch])
    
    ### preview graphs
    blocking = True if not flag_bycmdline else False
    
    x = np.arange(max_epochs+1)
    plt.figure()
    plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
    plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
    plt.xlabel("epochs k")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.savefig(dpath+"/accuracy"+suffix+suffix2+".png")
    plt.show(block=blocking)
    
    plt.figure()
    if not flag_online:
        x = np.arange(iter_per_epoch * (max_epochs-1) +1)
        plt.plot(x, train_loss_list, label='train', linewidth=1.0)
    else:
        x = np.arange(max_epochs)
        plt.plot(x, loss_each_list, label='train', linewidth=1.0)
    plt.xlabel("iterations")
    plt.ylabel("loss (mean squared error)")
    # plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.savefig(dpath+"/loss"+suffix+suffix2+".png")
    plt.show(block=blocking)

    z = np.array([agents[i].z_vec for i in range(n)])
    print('z:')
    print(z)
    W = np.array([agents[i].WeiG for i in range(n)])
    print('Weights:')
    print(W)

    return


def storeLossAndAccuracy(agents, n,
                         train_acc_list, test_acc_list, train_loss_list, test_loss_list,
                         disableTrain=False, disableTest=False):
    for i in range(n):
        agents[i].calcLoss(disableTrain,disableTest)
    
    if not disableTrain:
        train_acc_age = tuple(agents[i].train_acc for i in range(n))
        train_acc = np.mean(train_acc_age)
        train_acc_list.append(train_acc)
        
        train_loss_age = tuple(agents[i].train_loss for i in range(n))
        train_loss = np.mean(train_loss_age)
        train_loss_list.append(train_loss)
    
    if not disableTest:
        test_acc_age = tuple(agents[i].test_acc for i in range(n))
        test_acc = np.mean(test_acc_age)
        test_acc_list.append(test_acc)
        
        test_loss_age = tuple(agents[i].test_loss for i in range(n))
        test_loss = np.mean(test_loss_age)
        test_loss_list.append(test_loss)
    
    return train_acc, test_acc, train_loss, test_loss


def argparser():
    parser = ArgumentParser()
    parser.add_argument('-g', '--graph',
                        action='store',
                        default=type_Graph,
                        help='string: specify the communication graph shape ("complete", "balanced", "unbalanced", or "watts-strogatz")')
    parser.add_argument('-c', '--communicate', 
                        action="store_true",
                        default=False,
                        help='flag: communicate with neighbors')
    parser.add_argument('-o', '--overfit',
                        action='store_true',
                        default=False,
                        help='flag: narrow down the train datas')                
    parser.add_argument('-s', '--staticstep',
                        action='store_true',
                        default=False,
                        help='flag: use static step size')
    parser.add_argument('-O', '--online',
                        action='store_true',
                        default=False,
                        help='flag: use static step size')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    ### option があった場合のみflagを書き換える
    flag_bycmdline = False
    if len(sys.argv) > 1:
        args = argparser()
        type_Graph = args.graph or type_Graph
        flag_staticStep = args.staticstep
        flag_communicate = args.communicate
        flag_overfittest = args.overfit
        flag_bycmdline = True
    main(flag_bycmdline,type_Graph,flag_staticStep,flag_communicate,flag_overfittest,flag_online)
    