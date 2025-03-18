# Inspired by Lee et al's "Multi-way spectral partitioning and higher-order Cheeger inequalities"
# https://arxiv.org/abs/1111.1055
import networkx as nx
from gerrychain.partition import Partition
import numpy as np
from sklearn.cluster import KMeans
from scipy import linalg as la
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Callable, Any


def repartition(partition: Partition, G: nx.Graph, k: int, nodelist: list[Any]) -> Partition:
    '''
        Returns a new partition by embedding the vertices into a higher-dimensional space and
        partitioning them using radial decomposition.
    '''
    nodeindices = {n: i for i,n in enumerate(nodelist)}
    
    F = Embed_Vertices(G, k, nodelist, nodeindices)
    S = Radial_Decomposition(F, k, nodelist)
    
    return Construct_Partition(S, G, k, partition)

def Embed_Vertices(
        G: nx.Graph,
        k: int, nodelist: list,
        nodeindices: dict) -> Callable[[Any], np.ndarray]:
    '''
        Embeds the vertices of `G` into a `k`-dimensional space using the first `k` eigenvalues of
        `G`'s normalized laplacian.
    '''
    # (|V|, |V|) matrices
    L = nx.normalized_laplacian_matrix(G, nodelist).todense()
    pow_D_minusHalf = np.diag([1/np.sqrt(G.degree(v)) for v in nodelist]) # type: ignore
    
    # (|V|, k) matrices
    _, g = la.eigh(L, subset_by_index=[0, k - 1])
    f = pow_D_minusHalf @ g # double check
    
    return lambda v: f[nodeindices[v],...]

def Radial_Decomposition(F: Callable, k: int, nodelist: list) -> list[set]:
    '''
        Projects the embedded vertices onto the `k-1`-sphere, then clusters them using `k`-means.
    '''
    normalize = lambda x: x / np.linalg.norm(x)
    normalized_vertices = [normalize(F(v)) for v in nodelist]
    
    cluster_size = len(normalized_vertices)//k + 1 # +1 to make sure every node is considered
    partition = [set() for _ in range(k)]

    kmeans = KMeans(n_clusters=k, init="random", max_iter=1000, tol=1e-6).fit(normalized_vertices)

    # maps (k, *) centers array to an (n, *) array, s.t. each row is repeated n/k times
    centers = kmeans.cluster_centers_.repeat(cluster_size, 0)
    
    # Dist[i,j] = dist(vertex[i], center[j])
    distance_matrix = cdist(normalized_vertices, centers)
    
    # assign vertices evenly to parts s.t. distance is minimized
    clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size

    for i,part in enumerate(clusters):
        partition[part].add(nodelist[i])

    return partition

def Construct_Partition(S: list[set], G: nx.Graph, k: int, partition: Partition) -> Partition:
    '''
        Converts the clusters found during radial decomposition into a partition of `G`'s vertices.
    '''
    # TODO: make a better way of assigning missed nodes
    S.sort(key=lambda s: weight(s, G))
    parts = list(partition.parts.keys())
    assert len(parts) == k, f'{len(parts)} != {k}'
    
    flips = {v: parts[k-1] for v in G.nodes}
    for i in range(k):
        for v in S[i]:
            flips[v] = parts[i]
            
    return partition.flip(flips)

def weight(S: set, G: nx.Graph) -> int:
    total = 0
    for v in S:
        total += G.degree(v) # type: ignore
    return total