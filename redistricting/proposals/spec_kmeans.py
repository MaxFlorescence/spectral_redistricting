'''Baseline Spectral K-Means clustering proposal for use with GerryChain.'''
# Inspired by Lee et al's "Multi-way spectral partitioning and higher-order Cheeger inequalities"
# https://arxiv.org/abs/1111.1055

from typing import Callable, Any

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from scipy import linalg as la
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from gerrychain.partition import Partition

def repartition(partition: Partition, graph: nx.Graph, k: int, nodelist: list[Any]) -> Partition:
    '''
    Returns a new partition by embedding the vertices into a higher-dimensional space and
    partitioning them using radial decomposition.
    '''
    nodeindices = {n: i for i,n in enumerate(nodelist)}

    f = embed_vertices(graph, k, nodelist, nodeindices)
    s = radial_decomposition(f, k, nodelist)

    return construct_partition(s, graph, k, partition)

def embed_vertices(
        graph: nx.Graph,
        k: int, nodelist: list,
        nodeindices: dict) -> Callable[[Any], np.ndarray]:
    '''
    Embeds the vertices of `G` into a `k`-dimensional space using the first `k` eigenvalues of
    `G`'s normalized laplacian.
    '''
    # (|V|, |V|) matrices
    laplacian_normalized = nx.normalized_laplacian_matrix(graph, nodelist).todense()
    degrees_normalized = np.diag([1/np.sqrt(graph.degree(v)) for v in nodelist]) # type: ignore

    # (|V|, k) matrices
    _, g = la.eigh(laplacian_normalized, subset_by_index=[0, k - 1])
    f = degrees_normalized @ g # double check

    return lambda v: f[nodeindices[v],...]

def normalize(x: Any) -> Any:
    '''
    Normalize a numpy array
    '''
    return x / np.linalg.norm(x)

def radial_decomposition(f: Callable, k: int, nodelist: list) -> list[set]:
    '''
    Projects the embedded vertices onto the `k-1`-sphere, then clusters them using `k`-means.
    '''
    normalized_vertices = [normalize(f(v)) for v in nodelist]

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

def construct_partition(s: list[set], graph: nx.Graph, k: int, partition: Partition) -> Partition:
    '''
    Converts the clusters found during radial decomposition into a partition of `G`'s vertices.
    '''
    # TODO: make a better way of assigning missed nodes
    s.sort(key=lambda s: weight(s, graph))
    parts = list(partition.parts.keys())
    assert len(parts) == k, f'{len(parts)} != {k}'

    flips = {v: parts[k-1] for v in graph.nodes}
    for i in range(k):
        for v in s[i]:
            flips[v] = parts[i]

    return partition.flip(flips)

def weight(s: set, graph: nx.Graph) -> int:
    '''
    Return the total weight of all vertices in `s`.
    '''
    total = 0
    for v in s:
        total += graph.degree(v) # type: ignore
    return total
