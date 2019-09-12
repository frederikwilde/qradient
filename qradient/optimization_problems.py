import numpy as np

class MaxCut:
    def __init__(self, vertex_num, edge_num=None, edge_set=None):
        all_edges = []
        for i in range(vertex_num):
            for j in range(i+1, vertex_num):
                all_edges.append([i, j])
        self.all_edges = np.array(all_edges)
        self.vertex_num = vertex_num
        if edge_set != None:
            if edge_num != None and edge_num != len(edge_set):
                raise ValueError('edge_set has length {}, but edge_num was specified as {}.'.format(len(edge_set), edge_num))
            self.edge_set = edge_set
            self.edge_num = len(edge_set)
        elif edge_num != None:
            self.edge_num = edge_num
            self.random()
        else:
            raise ValueError('Specify either edge_set or edge_num')

    def random(self):
        indeces = np.random.choice(len(self.all_edges), size=self.edge_num, replace=False)
        edge_set = []
        for i in range(self.edge_num):
            edge = self.all_edges[indeces[i]]
            edge_set.append(edge)
        self.edge_set = np.array(edge_set)

    def to_observable(self):
        zz = np.full([self.vertex_num, self.vertex_num], None)
        for edge in self.edge_set:
            zz[edge[0], edge[1]] = 1.
        return {'zz': zz}

    def plot_lists(self):
        vertex_locations = []
        vertex_loc_tuples = []
        for i in range(self.vertex_num):
            vertex_locations.append([np.sin(2*np.pi*i/self.vertex_num)])
            vertex_locations.append([np.cos(2*np.pi*i/self.vertex_num)])
            vertex_loc_tuples.append([vertex_locations[-2], vertex_locations[-1]])
            vertex_locations.append('ro')
        edge_locations = []
        for edge in self.edge_set:
            v1 = vertex_loc_tuples[edge[0]]
            v2 = vertex_loc_tuples[edge[1]]
            edge_locations.append([v1[0], v2[0]])
            edge_locations.append([v1[1], v2[1]])
            edge_locations.append('black')
        return vertex_locations, edge_locations
