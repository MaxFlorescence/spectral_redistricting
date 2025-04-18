'''Updater methods for collecting partition metrics.'''

from typing import Any
from functools import lru_cache, partial
import inspect

import networkx as nx
import numpy as np

from gerrychain.partition import Partition
from gerrychain.updaters import cut_edges

def tally_population(part: Any, partition: Partition) -> float:
    '''
    Compute the population of one `part` in the `partition`.
    '''
    return sum(partition.graph.nodes[n]['pop'] for n in part)

class Metrics:
    '''
    A class for providing updater functions.
    '''
    def __init__(self, target_size: float, target_population: float):
        assert target_size != 0
        self.target_size = target_size
        assert target_population != 0
        self.target_population = target_population

    @lru_cache(maxsize=1)
    def __extreme_parts(self, partition: Partition, use_population: bool) -> tuple[int, int]:
        '''
        Returns the sizes of the largest and smallest parts of the partition.
        '''
        max_part, min_part = None, None

        for part in partition.parts.values():
            size = len(part) if not use_population else tally_population(part, partition)

            if max_part is None or max_part < size:
                max_part = size
            if min_part is None or min_part > size:
                min_part = size

        assert max_part is not None and min_part is not None
        return max_part, min_part

    def cut_edges(self, partition: Partition) -> list[tuple]:
        '''
        Just calls `gerrychain.updaters.cut_edges` but returns a list instead of a set.
        '''
        return list(cut_edges(partition))

    def cut_edge_count(self, partition: Partition) -> int:
        '''
        Computes the number of edges cut by the `partition`.
        '''
        return len(cut_edges(partition))

    def __disparity(self, partition: Partition, use_population: bool) -> float:
        '''
        Returns the ratio in size/population of the largest and smallest parts. 
        '''
        max_part, min_part = self.__extreme_parts(partition, use_population)
        if min_part == 0:
            min_part = 1

        return max_part/min_part

    def size_disparity(self, partition: Partition):
        '''
        Computes the maximum size disparity of the `partition`.
        
        This equals the size of the largest part divided by the size of the smallest part.
        '''
        return self.__disparity(partition, use_population=False)

    def population_disparity(self, partition: Partition):
        '''
        Computes the maximum population disparity of the `partition`.
        
        This equals the population of the largest part divided by the population of the smallest
        part.
        '''
        return self.__disparity(partition, use_population=True)

    def __deviation(self, partition: Partition, use_population: bool) -> float:
        '''
        Computes the deviation from target of the `partition`.
        
        This equals the maximum distance of any part from the target divided by the ideal.
        '''
        target = self.target_population if use_population else self.target_size

        parts = self.__extreme_parts(partition, use_population)
        return max(abs(p - target) for p in parts) / target

    def size_deviation(self, partition: Partition):
        '''
        Computes the deviation from the ideal part sizes of the `partition`.
        
        This equals the maximum distance of any part from the ideal size divided by the ideal size.
        '''
        return self.__deviation(partition, use_population=False)

    def population_deviation(self, partition: Partition):
        '''
        Computes the deviation from the ideal part populations of the `partition`.
        
        This equals the maximum distance of any part from the ideal population divided by the ideal
        population.
        '''
        return self.__deviation(partition, use_population=True)

    def contiguous(self, partition: Partition):
        '''
        Implementation of `gerrychain.constraints.contiguous` since that seems to crash a lot.
        '''
        for part in partition.parts.values():
            if not nx.is_connected(partition.graph.subgraph(part)): # type: ignore
                return False

        return True

    def districts(self, partition: Partition, use_population: bool = False):
        '''
        Returns a list of each district's size.
        '''
        size = partial(tally_population, partition=partition) if use_population else len
        return [size(p) for p in partition.parts.values()]

    def __producing_spanning_trees(self, partition: Partition, log: bool) -> float|int:
        # Clelland et al's P_D for arbitrary k
        '''
        Counts the number of producing spanning trees of the `partition`.
        
        If `log` is true, returns the natural logarithm of the count.
        '''
        result = 0 if log else 1
        def accumulate(val: float) -> None:
            int_val = int(val + 0.5)

            nonlocal result
            if log:
                result += np.log(int_val)
            else:
                result *= int_val

        # number of spanning trees in each part
        for part in partition.parts:
            subgraph = partition.graph.subgraph(part)
            accumulate(nx.total_spanning_tree_weight(subgraph, weight=None))

        # number of ways to split a full spanning tree into the parts
        part_graph = nx.MultiGraph([
            (partition.assignment[u], partition.assignment[v])
            for u,v in filter(partition.crosses_parts, partition.graph.edges)
        ])
        accumulate(nx.total_spanning_tree_weight(part_graph, weight=None))

        return result

    def producing_spanning_trees(self, partition: Partition) -> float:
        '''
        Counts the number of producing spanning trees of the `partition`.
        
        A "producing spanning tree" of a k-partition is a spanning tree of the partition's parent
        graph for which cutting exactly k-1 edges results in the partition.
        '''
        return self.__producing_spanning_trees(partition, log=True)

    def log_producing_spanning_trees(self, partition: Partition) -> float:
        '''
        Returns the natural logarithm of the number of producing spanning trees of the `partition`.
        
        A "producing spanning tree" of a k-partition is a spanning tree of the partition's parent
        graph for which cutting exactly k-1 edges results in the partition.
        '''
        return self.__producing_spanning_trees(partition, log=True)

available_updaters = [
    member[0]
    for member in inspect.getmembers(Metrics, predicate=inspect.isfunction)
    if member[0][0] != '_'
]
