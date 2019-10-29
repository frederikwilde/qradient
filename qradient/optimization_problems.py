import numpy as np

class MaxCut:
    def __init__(self, vertex_num, **kwargs):
        all_edges = []
        for i in range(vertex_num):
            for j in range(i+1, vertex_num):
                all_edges.append([i, j])
        self.all_edges = np.array(all_edges)
        self.vertex_num = vertex_num
        if 'edge_set' in kwargs:
            self.edge_set = kwargs['edge_set']
        elif 'edge_num' in kwargs:
            self.__random(edge_num=kwargs['edge_num'])
        elif 'edge_probability' in kwargs:
            self.__random(edge_probability=kwargs['edge_probability'])
        else:
            raise ValueError('Specify one of the three edge_set, edge_num, or edge_probability')

    def __random(self, **kwargs):
        if 'edge_num' in kwargs:
            indeces = np.random.choice(range(len(self.all_edges)), size=kwargs['edge_num'], replace=False)
            self.edge_set = self.all_edges[indeces]
        elif 'edge_probability' in kwargs:
            self.edge_set = []
            for e in self.all_edges:
                if np.random.rand() < kwargs['edge_probability']:
                    self.edge_set.append(e)
            self.edge_set = np.array(self.edge_set)
        else:
            raise ValueError('Specify either edge_num or edge_probability.')

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
