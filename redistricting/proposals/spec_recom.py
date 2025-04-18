'''Spectral Recombination proposal for use with GerryChain.'''

from typing import Any, Literal
import random

import networkx as nx
from scipy import linalg as la

from gerrychain.partition import Partition

def repartition(
        partition: Partition,
        threshold: float|Literal["brute force"] = 0.0
    ) -> Partition:
    '''
    Returns a new partition by randomly recombining two distrcits and splitting them based on
    the combined district's Fiedler vector.
    ### Parameters
    - **partition** (*Partition*): The partition on which to perform a recombination step.
    - **threshold** (*float*|`"brute force"`): The cutoff in the Fiedler vector at which to
    split the combined districts. If the string `"brute force"` is given, then the best
    threshold will be calculated by considering each possibility.
    '''
    # Choose a random cut edge. partition['cut_edges'] is the set of edges that are cut by the
    # partition, since random.choice() needs an index-able data structure, we convert it to a tuple
    # first.
    cut_edge: tuple = random.choice(tuple(partition['cut_edges']))

    # Store the part indices that each endpoint of the two edges are in
    parts_to_merge = (partition.assignment[cut_edge[0]], partition.assignment[cut_edge[1]])

    # Merge the parts and get the subgraph on the merged nodes
    merged_nodes = [v for v in partition.graph.nodes if partition.assignment[v] in parts_to_merge]
    merged_subgraph = partition.graph.subgraph(merged_nodes).copy() # NEED to make a new graph

    # Tell the chain to flip part indices to match the clusters obained above
    flips = fiedler_cut(merged_subgraph, threshold, parts_to_merge, randomize_weights=True)
    return partition.flip(flips)

def fiedler_cut(subgraph: nx.Graph,
                threshold: float|str,
                part_labels: tuple = (0, 1),
                randomize_weights: bool = False,
                normalize_laplacian: bool = False) -> dict:
    '''
    Assigns nodes in the subgraph to parts based on the sign of their corresponding value in the
    fiedler vector.
    '''
    node_list = list(subgraph.nodes)
    n = len(node_list)

    _, fv = fiedler(subgraph, randomize_weights, normalize_laplacian)

    if threshold == 'brute force':
        threshold = threshold_sweep(fv, subgraph, node_list, n)

    assert isinstance(threshold, (float, int))

    # the partition is given by whether fv[x] >= threshold
    return {node_list[x]: part_labels[int(fv[x] >= threshold)] for x in range(n)}

def fiedler(graph: nx.Graph,
            randomize_weights: bool = False,
            normalize_laplacian: bool = False) -> tuple[float, Any]:
    '''
    Calculates the feidler value and vector for the given graph.
    '''
    if randomize_weights:
        # Make edge weight uniform random reals in the interval [0,1)
        for edge in graph.edges:
            graph.edges[edge]["weight"] = random.random() + 1

    # The Laplacian matrix of a graph is L = D - A, where A is the adjacency matrix and
    #   D is the diagonal matrix of node degrees.
    if normalize_laplacian:
        m = nx.normalized_laplacian_matrix(graph).todense()
    else:
        m = nx.laplacian_matrix(graph).todense()

    # Compute the eigenvectors corresponding to the first two eigenvalues
    eigvals, eigvecs = la.eigh(m, subset_by_index=[0,1])

    # The Fiedler vector is the one corresponding to the second eigenvalue, which
    #   due to 0-based indexing is eigenvectors[:, 1]
    fval = eigvals[1]
    fvec = eigvecs[:, 1]

    return fval, fvec

def threshold_sweep(fiedler_vector: Any,
                    subgraph: nx.Graph,
                    node_list: list[Any],
                    n: int) -> float:
    '''
    Consider all unique cuts according to the Fiedler vector and save the most balanced,
    connected one.
    '''
    minimum_split = [] # maintains minimum population difference, threshold, and parts
    for t in fiedler_vector:
        parts, diff = split_nodes(fiedler_vector, t, subgraph, node_list, n)
        if parts is None:
            # the cut produced one empty part
            continue
        s = {"diff": diff, "t": t, "parts": parts}

        if len(minimum_split) == 0:
            minimum_split.append(s)
            continue

        if minimum_split[0]["diff"] >= diff and is_connected(subgraph, parts):
            if minimum_split[0]["diff"] == diff:
                # split is tied with previous minimums
                minimum_split.append(s)
            else:
                # split is a new minimum
                minimum_split = [s]
    assert len(minimum_split) > 0

    # break ties with cut edge counts
    if len(minimum_split) > 1:
        minimum_split = min(minimum_split, key = lambda split: cut_edges(subgraph, split["parts"]))
    else:
        minimum_split = minimum_split[0]

    return minimum_split["t"]

def split_nodes(fiedler_vector: Any,
                threshold: float,
                graph: nx.Graph,
                node_list: list[Any],
                n: int) -> tuple[list[list[Any]]|None, int]:
    '''
    Returns the graph Fiedler-cut defined by the given threshold and the corresponding
    difference in population.

    Returns None,0 if one of the parts is the empty graph.
    '''
    # split nodes
    node_split = [{"nodes":[], "sign":1}, {"nodes":[], "sign":-1}]
    for i,n in enumerate(node_list):
        part = int(fiedler_vector[i] >= threshold)
        node_split[part]["nodes"].append(n)

    # check for trivial split
    if len(node_split[0]["nodes"]) in [0, n]:
        return None, 0

    # tally population difference
    diff = 0
    for part in node_split:
        diff += part["sign"]*sum(graph.nodes[n]['pop'] for n in part["nodes"])

    return [part["nodes"] for part in node_split], abs(diff)

def is_connected(graph: nx.Graph, parts: list[list[Any]]) -> bool:
    '''
    Returns True iff all parts correspond to connected subgraphs.
    '''
    for part in parts:
        # latest possible point to induce subgraphs
        if not nx.is_connected(nx.induced_subgraph(graph, part)):
            return False
    return True

def cut_edges(graph: nx.Graph, parts: list[list[Any]]) -> int:
    '''
    Count edges cut by the parts.
    '''
    part_map = {}
    for i,p in enumerate(parts):
        part_map.update({n: i for n in p})

    subgraph = nx.induced_subgraph(graph, part_map.keys())

    count = 0
    for e in subgraph.edges:
        if part_map[e[0]] != part_map[e[1]]:
            count += 1

    return count
