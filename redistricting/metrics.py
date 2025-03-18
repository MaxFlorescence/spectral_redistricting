from gerrychain.partition import Partition
from gerrychain.updaters import cut_edges
import networkx as nx
import numpy as np
from typing import Callable, Any

def cut_edge_count(partition: Partition) -> int:
    return len(cut_edges(partition))
            
def disparity(partition: Partition, use_population: bool = False) -> float:
    '''
    Computes the maximum disparity of the `partition`.
    
    This equals the size of the largest part divided by the size of the smallest part.
    '''
    max_part, min_part = extreme_parts(partition, use_population)
    if min_part == 0:
        min_part = 1
    
    return max_part/min_part

def deviation(partition: Partition, target: float, use_population: bool = False) -> float:
    '''
    Computes the deviation from target of the `partition`.
    
    This equals the maximum distance of any part from the target divided by the ideal.
    '''
    assert target != 0
    
    parts = extreme_parts(partition, use_population)
    return max(abs(p - target) for p in parts) / target
    
def extreme_parts(partition: Partition, use_population: bool = False) -> tuple[int, int]:
    '''
    Returns the sizes of the largest and smallest parts of the partition.
    '''
    max_part, min_part = None, None
    
    for part in partition.parts.values():
        size = len(part) if not use_population else sum(
            partition.graph.nodes[n]['pop'] for n in part
        )
        
        if max_part is None or max_part < size:
            max_part = size
        if min_part is None or min_part > size:
            min_part = size

    assert max_part is not None and min_part is not None
    return max_part, min_part

def contiguous(partition: Partition):
    '''
    Implementation of `gerrychain.constraints.contiguous` since that seems to crash a lot.
    '''
    for part in partition.parts.values():
        if not nx.is_connected(partition.graph.subgraph(part)): # type: ignore
            return False
        
    return True

def districts(partition: Partition, use_population: bool = False):
    '''
        List each district's size.
    '''
    size = (lambda p: sum(partition.graph.nodes[n]['pop'] for n in p)) if use_population else len
    return [size(p) for p in partition.parts.values()]

def producing_spanning_trees(partition: Partition, log: bool = True) -> float|int:
    # Clelland et al's P_D for arbitrary k
    '''
    Counts the number of producing spanning trees of the `partition`.
    
    A "producing spanning tree" of a k-partition is a spanning tree of the partition's parent graph
    for which cutting exactly k-1 edges results in the partition.
    
    If `log` is true, returns the natural logarithm of the count.
    '''
    factors = []
    
    # number of spanning trees in each part
    for part in partition.parts:
        subgraph = partition.graph.subgraph(part)
        T = my_round(nx.total_spanning_tree_weight(subgraph, weight=None)) # type: ignore
        factors.append(T)
    
    # number of ways to split a full spanning tree into the parts
    part_graph = nx.MultiGraph([
        (partition.assignment[u], partition.assignment[v])
        for u,v in filter(partition.crosses_parts, partition.graph.edges)
    ])
    T = my_round(nx.total_spanning_tree_weight(part_graph, weight=None))
    factors.append(T)
    
    if log:
        return np.sum(np.log(factors), dtype=float)
    else:
        return np.prod(factors, dtype=int)

def my_round(n: float) -> int:
    return int(n + 0.5)

def get_updater(name: str,
                   target_size: float,
                   target_population: float) -> Callable[[Partition], Any]:
    assert name in updater_map, f'Unknown updater "{name}"!'
        
    t = 0
    if name == "size deviation":
        t = target_size
    if name == "population deviation":
        t = target_population
        
    return updater_map[name](t)

updater_map = { # kinda gross looking, oh well
    'cut edges':                    lambda _: cut_edge_count,
    'size disparity':               lambda _: disparity,
    'population disparity':         lambda _: lambda p: disparity(p, use_population=True),
    'size deviation':               lambda t: lambda p: deviation(p, t),
    'population deviation':         lambda t: lambda p: deviation(p, t, use_population=True),
    'contiguous':                   lambda _: contiguous,
    'district sizes':               lambda _: districts,
    'district populations':         lambda _: lambda p: districts(p, use_population=True),
    'producing spanning trees':     lambda _: lambda p: producing_spanning_trees(p, log=False),
    'log producing spanning trees': lambda _: producing_spanning_trees
}

available_metrics = tuple(updater_map.keys())