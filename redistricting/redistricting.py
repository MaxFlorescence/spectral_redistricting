import os, json, time
import networkx as nx
from typing import Any, Literal, Callable
from redistricting.proposals import (spectral_kmeans, spectral_recom)
from redistricting import (metrics, utils)
from gerrychain import MarkovChain
from gerrychain.accept import always_accept
from gerrychain.constraints import (Validator, within_percent_of_ideal_population)
from gerrychain.partition import Partition
from gerrychain.proposals import recom, reversible_recom
from gerrychain.updaters import Tally

class Redistricting:
    '''
        A wrapper class for [GerryChain](https://github.com/mggg/GerryChain) that makes it simpler
        to run redistricting Markov chains on different combinations of parameters.
        
        The redistricting problem takes as input a graph partitioned into k districts and produces
        as output a new partition. Using Markov chains, the problem can be solved by iterating on
        the starting partition, where a new partition is proposed at each step.
        
        ### Instance Methods:
        - `run()`: Run the instance and collect data on it.
        
        ### Class Methods:
        - `__init__()`: Create a new instance to run.
        - `from_checkpoint()`: Use a file to create an instance that, when run, resumes a previously
        interrupted instance.
    '''

    @classmethod
    def from_checkpoint(cls,
                        json_file: str) -> 'tuple[bool, Redistricting|None]':
        '''
            Create and return a Redistricting object using information in a json file.
            When run, the object will resume execution and data collection from where the checkpoint
            specifies.
            
            ### Parameters
            - **json_file** (*str*): The path to the .json file from which to load the checkpoint.
            If the '.json' extension is not at the end of this path, it will be appended.
            
            ### Returns
            **(status, object)** pair:
            - **status** is False if the run associated with the json file had already completed,
            and True if the file was not found or if the run is unfinished.
            - **object** is not None iff the json file represents an unfinished Redistricting run.
            
            If **status** is False, then **object** is None.
            
            ### Example Usage
            ```
            # making checkpoints
            r = Redistricting(...)
            r.run(
                ...,
                checkpoint_interval=5,
                checkpoint_dest='my_checkpoint'
            )
            
            # using checkpoints
            ok, r = Redistricting.from_checkpoint('my_checkpoint')
            if ok and r is not None:
                r.run(...)
            ```
        '''
        if json_file[-5:] != '.json':
            json_file += '.json'
        
        json_file = os.path.abspath(json_file)
        if not os.path.exists(json_file):
            return True, None
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            if 'finished' in data and data['finished']:
                return False, None
            
            node_type = utils.type_map[data['node_type']]
            
            new_constructor_data = data['constructor'].copy()
            new_constructor_data['steps'] = data['steps']
            new_constructor_data['assignment'] = data['assignment'] if node_type == str else {
                node_type(n): d
                for n,d in data['assignment'].items()
            }
            
            r = Redistricting(**new_constructor_data)
            r.__set_constructor_data(data['constructor'])
            return True, r
    
    def __init__(self,
                 graph: nx.Graph|Literal['grid', 'triangular']|str,
                 k: int,
                 assignment: dict[str, int]|Literal['row', 'col']|str,
                 proposal: Literal[
                     'identity', 'recom', 'revrecom', 'specrecom', 'bspecrecom', 'speckmeans'
                 ],
                 steps: int,
                 step_updaters: dict[str, Callable[[Partition], Any]]|list[str] = [],
                 single_updaters: dict[str, Callable[[Partition], Any]]|list[str] = [],
                 population_key: str|None = 'pop',
                 h: int = 0,
                 w: int = 0,
                 graph_name: str|None = None,
                 assignment_name: str|None = None) -> None:
        '''
            Creates a Redistricting object that can then be run.
            
            ### Parameters
            - **graph** (*Graph* | *str*): The graph to partition. Values of 'grid' and 'triangular'
            create a new grid or triangular lattice graph of size n x m. Other string values are
            interpreted as file names from which to load a graph using GerryChain's
            `Graph.from_file()` method. The node's identifiers are converted to strings.
            
            - **k** (*int*): The number of districts that the graph should be partitioned into.
            
            - **assignment** (*dict* | *str*): The starting partition's assignment of nodes to
            district IDs. Values of 'row' and 'col' create a new assignment where the graph's nodes
            are split into k rows or columns of approximately equal size.
            
            - **proposal** (*str*): The algorithm to run at each step of the chain. The algorithms
            'identity' and 'speckmeans' are not Markov chains, and so they only run for one step
            (ignoring the value of the steps parameter).
            
            - **steps** (*int*): How many steps to run the markov chain for. A step is a transition
            from one state to another, so after running there will have been steps+1 states. A value
            of zero will force the proposal to be the identity function.
            
            - **step_updaters** (*dict* | *list*): Functions to run on each step's partition to
            measure its attributes. See `redistricting.available_metrics` for a list of acceptable
            strings.
            
            - **single_updaters** (*dict* | *list*): Functions to run only on the first and last
            partitions to measure their attributes. See `redistricting.available_metrics` for a
            list of acceptable strings.
            
            - **population_key** (*str*): The key of the attribute containing each node's
            population. Ignored if graph *is* 'grid' or 'triangular'. Any value except for 'pop'
            will create a new attribute with key 'pop', into which the node populations will be
            copied. A value of None will assign population 1 to each node (this will also happen for
            a value of 'pop', for each node with no 'pop' attribute).
            
            - **h** (*int*): The height of the graph. Ignored if graph *is not* 'grid' nor
            'triangular'.
            
            - **w** (*int*): The width of the graph. Ignored if graph *is not* 'grid' nor
            'triangular'.
            
            - **graph_name** (*str*): A name for the graph. A value of None will generate a name
            automatically.
            
            - **assignment_name** (*str*): A name for the assignment. A value of None will generate
            a name automatically.
            
            ### Example Usage
            ```
            # 12x12 grid into 4 pieces, tracking population deviation
            r = Redistricting(
                graph='grid',
                k=4,
                assignment='row',
                proposal='recom',
                steps=10,
                step_updaters=['population deviation'],
                h=12,
                w=12
            )
            r.run()
            ```
        '''
        # store data for checkpointing
        self.instance_data = {
            'checkpoint': {
                'constructor': {
                    'k': k,
                    'assignment': assignment,
                    'proposal': proposal,
                    'steps': steps,
                    'population_key': population_key,
                    'h': h,
                    'w': w,
                    'graph_name': graph_name,
                    'assignment_name': assignment_name
                },
                'steps': steps, # number of steps remaining
                'assignment': None # assignment at the time of checkpointing
            }
        }
        if not isinstance(graph, nx.Graph):
            self.instance_data['checkpoint']['constructor']['graph'] = graph
        if not isinstance(step_updaters, dict):
            self.instance_data['checkpoint']['constructor']['step_updaters'] = step_updaters
        if not isinstance(single_updaters, dict):
            self.instance_data['checkpoint']['constructor']['single_updaters'] = single_updaters
            
        if steps == 0:
            proposal = 'identity'
        
        # the rest of the initialization
        self.__init_names(graph_name, graph, h, w, assignment_name, assignment, proposal)
        
        self.k = k
        self.__init_graph(graph, population_key, h, w)
        
        # population updater is required
        self.step_updaters = self.__init_updaters(step_updaters) | {"population": Tally("pop")}
        self.single_updaters = self.__init_updaters(single_updaters)
        
        self.steps = max(steps, 1)
        self.__init_chain(assignment, proposal)
        
        self.__store_runtime_data()
        
    def __init_names(self,
                       graph_name: str|None,
                       graph: nx.Graph|str,
                       h: int,
                       w: int,
                       assignment_name: str|None,
                       assignment: dict[Any, int]|str,
                       proposal: str) -> None:
        '''
            Determine what graph_name/assignment_name should be if it's None. Sets the instance
            variables:
            - self.graph_name
            - self.assignment_name
            - self.proposal_name
        '''
        if graph_name is None:
            if graph in ['grid', 'triangular']:
                self.graph_name = f'{h}x{w} {graph}'
            elif isinstance(graph, str):
                self.graph_name = graph
            else:
                self.graph_name = 'custom'
        else:
            self.graph_name = graph_name
            
        self.instance_data['checkpoint']['constructor']['graph_name'] = self.graph_name
            
        if assignment_name is None:
            if isinstance(assignment, str):
                self.assignment_name = assignment
            else:
                self.assignment_name = 'custom'
        else:
            self.assignment_name = assignment_name
            
        self.instance_data['checkpoint']['constructor']['assignment_name'] = self.assignment_name
            
        self.proposal_name = proposal

    def __init_graph(self,
                     graph: nx.Graph|str,
                     population_key: str|None,
                     h: int,
                     w: int) -> None:
        '''
            Initialize the graph and its nodes, and set the instance variables:
            - self.graph
            - self.graph_is_custom
            - self.nodelist
            - self.N
            - self.target_size
        '''
        self.graph, self.graph_is_custom = self.__load_graph(graph, h, w)
        
        self.nodelist = list(self.graph.nodes)
        self.N = len(self.nodelist)
        self.target_size = self.N / self.k
        
        node_type = str(type(self.nodelist[0]))
        if node_type not in utils.type_map:
            print(f'Warning: nodes of type "{node_type}" are not supported by checkpointing.')
        self.instance_data['checkpoint']['node_type'] = node_type
        
        self.__set_populations(self.graph, population_key)
    
    def __load_graph(self,
                     graph: nx.Graph|str,
                     h: int,
                     w: int) -> tuple[nx.Graph, bool]:
        '''
            Load the graph from utils or a file if it's a string, and set the instance variables:
            - self.node_size
            - self.node_shape
        '''
        self.node_size = 0
        self.node_shape = ''
        is_custom = True
        
        if isinstance(graph, str):
            if graph == 'grid':
                graph, self.node_size = utils.grid_graph(h, w)
                self.node_shape = 's'
                is_custom = False
            elif graph == 'triangular':
                graph, self.node_size = utils.triangular_graph(h, w)
                self.node_shape = 'h'
                is_custom = False
            elif os.path.exists(graph):
                graph = utils.graph_from_file(graph)
            else:
                raise Exception(f'The graph "{graph}" doesn\'t exist!')
            
        return graph, is_custom
    
    def __set_populations(self,
                          graph: nx.Graph,
                          population_key: str|None) -> None:
        '''
            Make sure the graph's nodes each have a 'pop' attribute, and set the instance variables:
            - self.total_population
            - self.target_population
        '''
        self.total_population = self.N
        
        if self.graph_is_custom:
            self.total_population = 0
            
            for n in graph.nodes:
                if population_key is None:
                    graph.nodes[n]['pop'] = 1
                elif population_key != 'pop':
                    graph.nodes[n]['pop'] = graph.nodes[n][population_key]
                elif 'pop' not in graph.nodes[n]: # and population_key == 'pop'
                    graph.nodes[n]['pop'] = 1
                    
                node_pop = graph.nodes[n]['pop']
                
                assert isinstance(node_pop, int), f'population_key attribute "{population_key}" of node "{n}" must be an integer!'
                
                self.total_population += node_pop
        
        self.target_population = self.total_population / self.k
    
    def __init_updaters(self,
                        updaters: dict[str, Callable[[Partition], Any]]|list[str]) -> dict[str, Callable[[Partition], Any]]:
        '''
            Create a dictionary of updater functions if updaters is a list of strings.
        '''
        if isinstance(updaters, list):
            return {name: metrics.get_updater(name, self.target_size, self.target_population) for name in updaters}
        else:
            return updaters
        
    def __init_chain(self,
                     assignment: dict[Any, int]|str,
                     proposal: str) -> None:
        '''
            Initialize the assignment, partition, and markov chain. Also set the instance variables:
            - self.assignment
            - self.near_target_population
            - self.chain
        '''
        self.assignment: dict[Any, int] = self.__init_assignment(assignment)
        
        partition = Partition(
            graph = self.graph,
            assignment = self.assignment,
            updaters = self.step_updaters
        )
        
        self.near_target_population = within_percent_of_ideal_population(partition, 0.1)
        
        # constraints seem to cause lots of rejections...
        ignore_constraints = True
        
        self.chain = MarkovChain(
            proposal = self.__init_proposal(proposal),
            constraints = Validator([] if ignore_constraints else [
                self.near_target_population,
                metrics.contiguous
            ]),
            accept = always_accept,
            initial_state = partition,
            total_steps = self.steps + 1 # counter-intuitively, the initial state counts as a step
        )
        
    def __init_assignment(self,
                          assignment: dict[str, int]|str) -> dict:
        '''
            Create an assignment mapping nodes to district IDs if assignment is a string.
        '''
        if assignment == "row":
            assignment = utils.stripes(
                self.nodelist, self.k, lambda n: self.graph.nodes[n]['pos'][1]
            )
        elif assignment == "col":
            assignment = utils.stripes(
                self.nodelist, self.k, lambda n: self.graph.nodes[n]['pos'][0]
            )
        elif assignment == "none":
            assert self.k == 1, "Cannot have a none assignment if k is greater than 1!"
            assignment = {n: 1 for n in self.nodelist}
        elif isinstance(assignment, str):
            assignment = nx.get_node_attributes(self.graph, assignment) # Trust the user
            assert isinstance(assignment, dict)
            
        return assignment
    
    def __init_proposal(self,
                        proposal: str) -> Callable[[Partition], Partition]:
        '''
            Creates the proposal function if proposal is a string. Also sets self.steps to 2 if
            necessary.
        '''
        proposal = proposal.lower()
        
        if proposal == "identity":
            proposal_fn = lambda p: p
            self.steps = 1
        elif proposal == 'recom':
            proposal_fn = lambda p: recom(
                partition = p,
                pop_col = 'pop',
                pop_target = self.target_population,
                epsilon = 0.05,
                region_surcharge = None
            )
        elif proposal == "revrecom":
            proposal_fn = lambda p: reversible_recom(
                partition = p,
                pop_col = 'pop',
                pop_target = self.target_population,
                repeat_until_valid = True,
                epsilon = 0.01, # Cannon et al. "Spanning Tree Methods for Sampling\
                M = 30          # Graph Partitions", table 1
            )
        elif proposal == "speckmeans":
            proposal_fn = lambda p: spectral_kmeans(
                partition = p,
                G = self.graph,
                k = self.k,
                nodelist = self.nodelist
            )
            self.steps = 1
        elif proposal == "specrecom": 
            proposal_fn = spectral_recom
        elif proposal == "balspecrecom":
            proposal_fn = lambda p: spectral_recom(
                partition = p,
                threshold = "brute force"
            )
        else:
            raise Exception(f'Unknown proposal "{proposal}"!')
        
        return proposal_fn
    
    def __set_constructor_data(self,
                           constructor_data: dict[str, Any]) -> None:
        '''
            Overwrites the constructor data in the instance_data dictionary.
        '''
        self.instance_data['checkpoint']['constructor'] = constructor_data
        
    def __store_runtime_data(self) -> None:
        '''
            Stores data generated after construction into the instance_data dictionary.
        '''
        self.instance_data['runtime'] = {
            'graph_is_custom': self.graph_is_custom,
            'N': self.N,
            'edges': len(self.graph.edges),
            'total_population': self.total_population
        }
        
        if self.steps != self.instance_data['checkpoint']['constructor']['steps']:
            self.instance_data['runtime']['steps'] = self.steps
        if self.graph_name != self.instance_data['checkpoint']['constructor']['graph_name']:
            self.instance_data['runtime']['graph_name'] = self.graph_name
        if self.assignment_name != self.instance_data['checkpoint']['constructor']['assignment_name']:
            self.instance_data['runtime']['assignment_name'] = self.assignment_name
    
    def __collect_single_updaters(self,
                                  partition: Partition|None,
                                  is_final: bool = True) -> None:
        '''
            Adds data generated by applying the single updaters to the partition to the
            instance_data dictionary.
        '''
        partition_type = 'last_partition' if is_final else 'first_partition'
        
        if partition is None:
            raise Exception(f"Cannot collect data on None {partition_type}!")
        
        self.instance_data['runtime'][partition_type] = {
            name: updater(partition) for name,updater in self.single_updaters.items()
        }
        
    def run(self,
            plot_interval: int = 1,
            interactive_level: Literal['script', 'progress', 'user', 'full'] = 'script',
            output_level: Literal['minimal', 'less', 'all'] = 'all',
            output_parent: str = '.',
            description: str|None = None,
            checkpoint_interval: int = 0,
            checkpoint_dest: str = 'checkpoint.json',
            keep_final_step: 'Callable[[dict[str, Any]], bool]|bool|None' = None) -> None:
        '''
            Runs the Redistricting Chain.
            
            ### Parameters
            - **plot_interval** (*int*): The interval at which the partitioned graphs will be
            plotted. The last step will always be plotted. If less than 1, then no steps will not be
            plotted (except the last step).
            - **interactive_level** (*str*): How to handle output.
                - 'script': Save run data and plots to the output_parent directory.
                - 'progress': Print progress info to the console, and save run data and plots to the
                  output_parent directory.
                - 'user': Print progress info and run data to the console, and display plots in real
                  time.
                - 'full': Do all of the above.
            - **output_level** (*str*): How much data to include in the output json file.
                - 'all': Include all parameters and runtime data.
                - 'less': Include only time elapsed and initial/final partition data.
                - 'minimal': The same as less, but also don't save any plots.
            - **output_parent** (*str*): The parent folder in which to save files to. Data and plots
            are saved to `output_parent/run/`, where the name `run` describes this run. Checkpoint
            files are saved directly to `output_parent/`.
            - **description** (*str*): A description of this run. This is incorporated into the
            `run` directory name, if not None.
            - **checkpoint_interval** (*int*): The interval at which checkpoints will be saved for
            this run. If less than 1, then no checkpoints will be made. Checkpoints can be resumed
            using `Redistricting.from_checkpoint()`.
            - **checkpoint_dest** (*str*): The file that checkpoints will be saved to. Only one
            checkpoint is maintained at a time.
            - **keep_final_step** (*(dict[str, Any]) -> bool*): A function whose output determines
            if the final step's partition data should be maintained in the checkpoint file. It will
            receive the output data dictionary as an argument. If False is returned (or the
            parameter is None), no data is kept in the checkpoint file.
        '''
        self.__parse_level(interactive_level)
        self.output_level = output_level
        self.plot_interval = plot_interval
        self.checkpoint_interval = checkpoint_interval
        if isinstance(keep_final_step, bool):
            self.keep_final_step = lambda _: keep_final_step
        else:
            self.keep_final_step = keep_final_step
        
        self.step_offset = self.instance_data['checkpoint']['constructor']['steps'] - self.steps
        if self.print_progress and self.step_offset > 0:
            print('Resuming from checkpoint.')
        
        description = self.__create_description(description)
        self.output_path, self.checkpoint_path = self.__setup_output_dirs(
            output_parent, description, checkpoint_dest
        )
        
        if self.print_progress:
            print(f'Running "{description}"...')
        
        self.instance_data['runtime']['time_elapsed'] = -time.time()
        initial_partition, final_partition = self.__run_the_chain()
        self.instance_data['runtime']['time_elapsed'] += time.time()
        
        if self.print_progress:
            print()
        
        run_data = self.__output_run_data(initial_partition, final_partition)
        
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r+') as f:
                output = {"finished": True}
                
                if self.keep_final_step is not None and self.keep_final_step(run_data):
                    # copy over existing checkpoint data
                    output.update(json.load(f))
                
                f.seek(0)
                f.truncate()    
                json.dump(output, f)
    
    def __create_description(self, description: str|None) -> str:
        '''
            Create a string to refer to this run by.
        '''
        self.instance_data['runtime']['description'] = description
        
        info = '{}_{}_{}_{}'.format(
            self.graph_name,
            self.k,
            self.proposal_name,
            self.instance_data['checkpoint']['constructor']['assignment_name']
        )
        
        if description is not None:
            info += '_' + description
            
        description = info.replace(' ', '-')

        return description
            
    def __parse_level(self, interactive_level: str) -> None:
        '''
            Maps interactive_level values to boolean variables.
        '''
        self.save_data = interactive_level != 'user'
        self.display_data = interactive_level in ['user', 'full']
        self.print_progress = interactive_level != 'script'
        
    def __setup_output_dirs(self,
                            output_parent: str,
                            description: str,
                            checkpoint_dest: str) -> tuple[str, str]:
        '''
            Make sure the output directories exist, if they're needed.
        '''
        output_path = os.path.join(output_parent, description)
        
        if checkpoint_dest[-5:] != '.json':
            checkpoint_dest += '.json'
        checkpoint_path = os.path.join(output_parent, checkpoint_dest)
        
        if not os.path.exists(output_parent):
            os.makedirs(output_parent, exist_ok=True) # for multiprocessing, need to create directory manually
        
        if self.save_data and self.output_level != "minimal" and not os.path.exists(output_path):
            os.mkdir(output_path)
            
        return output_path, checkpoint_path
    
    def __output_run_data(self,
                          initial_partition: Partition|None,
                          final_partition: Partition|None) -> dict[str, Any]:
        '''
            Collect info from the arguments and output this run's data.
        '''
        if self.plot_interval > 0:
            self.__collect_single_updaters(initial_partition, is_final=False)
        self.__collect_single_updaters(final_partition)
        
        run_data = {
            'parameters': self.instance_data['checkpoint']['constructor'],
            'runtime': self.instance_data['runtime']
        } if self.output_level == 'all' else {
            key: self.instance_data['checkpoint']['constructor'][key]
            for key in ['graph_name', 'k', 'proposal', 'assignment_name']
        } | {
            key: self.instance_data['runtime'][key]
            for key in [
                'description', 'last_partition', 'first_partition',
                'step_updater_data', 'time_elapsed'
            ] if key in self.instance_data['runtime']
        }

        if self.display_data:
            print(run_data)
        if self.save_data:
            file = os.path.join(self.output_path, "run_data.json")
            if self.output_level == "minimal":
                file = self.output_path + "_run-data.json"
            with open(file, "w") as f:
                json.dump(run_data, f, indent=4)
            if self.print_progress:
                print(f'Results saved to {file}')
        
        return run_data
    
    def __run_the_chain(self) -> tuple[Partition|None, Partition|None]:
        '''
            Run the main loop of the Markov chain, returning the first and last partitions generated.
        '''
        initial_partition = None
        final_partition = None
        
        self.instance_data['runtime']['step_updater_data'] = {
            updater: [] for updater in self.step_updaters.keys() if updater != "population"
        }
        
        for i, partition in enumerate(self.chain):
            assert partition is not None, f"None partition at step {i+self.step_offset}!"
            
            self.__try_checkpointing(i + self.step_offset, partition)
            
            if self.print_progress:
                print(f"\rstep {i+self.step_offset}/{self.steps+self.step_offset}", end='')
            
            if initial_partition is None:
                initial_partition = partition
            final_partition = partition
            
            if self.output_level != 'minimal' and self.__should_plot_step(i + self.step_offset):
                utils.draw_graph(
                    is_custom = self.graph_is_custom,
                    partition = partition,
                    graph = self.graph,
                    k = self.k,
                    node_size = self.node_size,
                    node_shape = self.node_shape,
                    show_graph = self.display_data,
                    file_name = os.path.join(
                        self.output_path, f"step_{i + self.step_offset}.png"
                    ) if self.save_data else None
                )
            
            for updater in self.step_updaters.keys():
                if updater != "population":
                    self.instance_data['runtime']['step_updater_data'][updater].append(
                        partition[updater]
                    )
                
        return initial_partition, final_partition
    
    def __try_checkpointing(self, i: int, partition: Partition) -> None:
        '''
            Try to make a checkpoint file for this step.
        '''
        steps = self.steps + self.step_offset
        
        if self.__should_checkpoint_step(i, steps):
            self.instance_data['checkpoint']['steps'] = steps - i
            self.instance_data['checkpoint']['assignment'] = dict(partition.assignment)
            with open(self.checkpoint_path, 'w') as f:
                json.dump(self.instance_data['checkpoint'], f)
                
    def __should_checkpoint_step(self, i: int, steps: int) -> bool:
        '''
            Should the checkpoint file be written to for step i?
        '''
        return (
            self.checkpoint_interval > 0
            and i not in [0, steps]
            and i % self.checkpoint_interval == 0
        ) or (self.keep_final_step is not None and i == steps)
    
    def __should_plot_step(self, i: int) -> bool:
        '''
            Should a matplotlib plot be generated for step i?
        '''
        steps = self.steps + self.step_offset
        return i == steps or (self.plot_interval > 0 and i % self.plot_interval == 0)
