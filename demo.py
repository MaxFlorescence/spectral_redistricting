from redistricting import Redistricting

checkpoint_file = "demo_checkpoint.json"
unfinished, r = Redistricting.from_checkpoint(checkpoint_file)
    
if unfinished:
    if r is None:
        # no checkpoint was found, so create the redistricting instance for the first time
        r = Redistricting(
            graph = "grid",                             # graph to redistrict
            k = 7,                                      # number of districts
            assignment = "row",                         # initial district assignment
            proposal = "specrecom",                     # proposal function for individual redistricting steps
            steps = 400,                                # number of steps to run the chain for
            step_updaters = ["cut edges"],              # statistics to collect after each step
            single_updaters = ["population deviation"], # statistics to collect at the start and at the end
            population_key = "pop",                     # key under which each node's population is stored
            h = 56,                                     # height of the graph (for grid and triangular graphs)
            w = 56,                                     # width of the graph (for grid and triangular graphs)
            graph_name = "56x56 grid graph",            # human-readable name of the graph
            assignment_name = "horizontal stripes"      # human-readable name of the assignment
        )
        
    r.run(
        plot_interval = 100,                            # how often to plot the graph (in intervals of steps)
        interactive_level = "progress",                 # how much information should be printed to the console
        output_level = "all",                           # how much information should be included in the output file
        output_parent = "./output/",                    # the directory to write output files and images to
        description = "grid graph run",                 # human-readable description of this run
        checkpoint_interval = 250,                      # how often to checkpoint the run (in intervals of steps)
        checkpoint_dest = checkpoint_file,              # the file to save checkpoints to
        keep_final_step = False                         # maintain the final step's partition data in the checkpoint file
    )
# else: the run associated with this checkpoint already finished