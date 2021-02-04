import dgl
from dgl.data import citation_graph as citegrh
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import itertools
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation


os.environ["DGLBACKEND"] = "pytorch"

def build_karate_club_graph():
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33])

    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
        29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
        31, 32])

    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])

    return dgl.graph((u, v))

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

def draw(i, nx_G, all_logits, all_acc, ax):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(34):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title(f'Epoch: {i}, Acc: {all_acc[i]:.3f}')
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
            with_labels=True, node_size=300, ax=ax)

if __name__ == "__main__":
    G = build_karate_club_graph()
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())

    nx_G = G.to_networkx().to_undirected()
    # pos = nx.kamada_kawai_layout(nx_G)
    # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    # plt.show()

    embed = nn.Embedding(34, 5)
    G.ndata['feat'] = embed.weight

    net = GCN(5, 5, 2)

    inputs = embed.weight

    # # semi-supervised clustering
    # labeled_nodes = torch.tensor([0, 33])
    # labels = torch.tensor([0, 1])

    # supervised:0 for instructor, 1 for president
    labeled_nodes = torch.tensor([i for i in range(34)])
    labels = torch.zeros_like(labeled_nodes)
    president_idxs = [14,15,18,20,22,26,29,23,24,25,27,28,31,9,32,33,30]
    labels[president_idxs] = 1

    optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
    all_logits = []
    all_acc = []
    for epoch in range(100):
        logits = net(G, inputs)
        all_logits.append(logits.detach())
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[labeled_nodes], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(logp.data, 1)
        correct_counnt = torch.sum(predicted==labels) 
        acc = correct_counnt.item()/len(labels)
        all_acc.append(acc)
        print(f'Epoch {epoch} | Loss: {loss.item():.4f} | ACC: {acc:.3f}')

    # visualize the process
    fig = plt.figure()
    ax = fig.subplots()
    draw_f = lambda i: draw(i, nx_G, all_logits, all_acc, ax)

    ani = animation.FuncAnimation(fig, draw_f, frames=len(all_logits), interval=200)
    plt.show()
