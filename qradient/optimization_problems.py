import numpy as np

class MaxCut:
    def __init__(self, vertex_num, edge_set=None, edge_num=None, edge_probability=None,):
        all_edges = []
        for i in range(vertex_num):
            for j in range(i+1, vertex_num):
                all_edges.append([i, j])
        self.all_edges = np.array(all_edges)
        self.vertex_num = vertex_num
        if edge_set != None:
            self.edge_set = edge_set
        elif edge_num != None or edge_probability != None:
            self.__random(edge_num=edge_num, edge_probability=edge_probability)
        else:
            raise ValueError('Specify one of the three edge_set, edge_num, or edge_probability')

    def __random(self, edge_num=None, edge_probability=None):
        if (edge_num == None and edge_probability == None) or (edge_num != None and edge_probability != None):
            # sanity check
            raise ValueError('Specify exactly one of two arguments.' + \
                f' edge_num was {edge_num} and edge_probability was {edge_probability}.')
        if edge_num != None:
            self.edge_set = np.random.choice(self.all_edges, size=edge_num, replace=False)
        else:
            self.edge_set = []
            for e in self.all_edges:
                if np.random.rand() < edge_probability:
                    self.edge_set.append(e)
            self.edge_set = np.array(self.edge_set)

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
