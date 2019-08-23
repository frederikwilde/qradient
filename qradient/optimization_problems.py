import numpy as np
import matplotlib.pyplot as plt

class MaxCut:
    def __init__(self, vertex_num, edge_num):
        all_edges = []
        for i in range(vertex_num):
            for j in range(i+1, vertex_num):
                all_edges.append([i, j])
        self.all_edges = np.array(all_edges)
        self.vertex_num = vertex_num
        self.edge_num = edge_num

    def random(self):
        indeces = np.random.choice(len(self.all_edges), size=self.edge_num, replace=False)
        edge_set = []
        for i in range(self.edge_num):
            edge = self.all_edges[indeces[i]]
            edge_set.append(edge)
        self.edge_set = np.array(edge_set)
        return self

    def to_observable(self):
        zz = np.full([self.vertex_num, self.vertex_num], None)
        for edge in self.edge_set:
            zz[edge[0], edge[1]] = 1.
        return {'zz': zz}

    def plot(self):
        vertex_locations = []
        for i in range(self.vertex_num):
            vertex_locations.append([np.sin(2*np.pi*i/self.vertex_num), np.cos(2*np.pi*i/self.vertex_num)])
        plt.figure(figsize=[4,4])
        plt.xticks([])
        plt.yticks([])
        for vl in vertex_locations:
            plt.plot(vl[0], vl[1], 'bo')
        for edge in self.edge_set:
            v1 = vertex_locations[edge[0]]
            v2 = vertex_locations[edge[1]]
            plt.plot([v1[0], v2[0]], [v1[1], v2[1]], 'black')
        plt.show()
