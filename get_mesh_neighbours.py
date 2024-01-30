import numpy as np 
import time 
from scipy.io import loadmat



from src import EIT


y_ref = loadmat('data/ref.mat') #load the reference data
Injref = y_ref["Injref"]

z0 = 1e-6* np.ones(32) #1./y0

solver = EIT(Injref, z0)


solver.omega.topology.create_connectivity(2, 1)
solver.omega.topology.create_connectivity(1, 2)


num_facets_owned_by_proc = solver.omega.topology.index_map(2).size_local

vertex_indices = np.arange(num_facets_owned_by_proc)
cell_connection_matrix = np.zeros((num_facets_owned_by_proc, num_facets_owned_by_proc), dtype=bool)

vertex_to_edge = solver.omega.topology.connectivity(2, 1)
edge_to_vertex = solver.omega.topology.connectivity(1, 2)

for vertex in vertex_indices:
    adjacent_edges = vertex_to_edge.links(vertex)
    for edge in adjacent_edges:
        adjacent_vertices = edge_to_vertex.links(edge)
        #print(adjacent_vertices)
        if len(adjacent_vertices) > 1:
            cell_connection_matrix[adjacent_vertices[0], adjacent_vertices[1]] = 1


np.save(f"data/mesh_neighbour_matrix.npy", cell_connection_matrix)

