// TODO: WHEN USING PAIR_IDS WITH CELL LIST FOR POINT PARTICLES, PAIR_IDS ARE NOT THE STATIC PARTICLE IDS!
// TODO: contact counting is broken!
// TODO: check speeds of stress tensor calculation for RB, i think it could be a lot faster
// TODO: make stress tensor (for poly particles) a vertex-level operation 
// TODO: define units
// TODO: define proper energy scale for rigid bumpy particles
// TODO: if rigid bumpy particle with core requires multiple energy scales, implement them
// TODO: add mixed-energy interaction case (compile option?) (is this something that can be handled by just scaling by the diameter?)
// TODO: implement hessian
// TODO: add heirarchical pre-req calculations!!!!!!
// TODO: make particle-agnostic loading method and scripts
// TODO: add console log
// TODO: add disk state reversion
// TODO: debate adding energy log
// TODO: fix bugs with memory access, probably due to improperly formatted input data
// TODO: add output manager configs and run argument configs - probably the same thing
// TODO: make pairwise interactions (for poly particles) a vertex-level operation - sum to particle level using vertex particle id comparison
// TODO: when using single vertex rigid bumpy particles, the cell list 2nd rebuild has a memory access error
// TODO: base particle can only have system-level output registry (non ordered) data
// TODO: point particle and poly particle should implement particle/vertex -level output registry (ordered) data
// TODO: rename cell aux -> cell write, add to base particle

// TODO: implement system averaging (like system segmented sum but then divided by system size)











// TODO: simplify the output manager
// TODO: add support for different logging schemes (linear, logarithmic)
// TODO: add energy file
// TODO: output manager configs





- Future Feature Idea – Trajectory Chunk Cache Budget
- Control: Expose a trajectory_chunk_cache_limit setting. During initialization compute the number of trajectory datasets (including the shared /timestep one). Derive a per-dataset budget limit / num_dsets and pass it to H5Pset_chunk_cache for every dataset access property list while creating/opening trajectory datasets.
- Motivation: Runs with millions of particles produce very wide chunks that overflow the 1 MB default cache, forcing HDF5 to flush chunks every timestep. A tunable budget keeps chunk writes in-memory longer, improving throughput, while still bounding peak RSS during large jobs.
- Behavior: HDF5 allocates chunk cache pages on demand up to the per-dataset limit. The global cap therefore remains “soft”: actual memory equals active_datasets × allocated_chunks. Document that users should scale the limit based on expected concurrency, and consider reducing budgets or closing datasets if many stay open concurrently.


// TODO: add a way to manually save an array to trajectory under the current step with optional array indices to save:

// TODO: redesign all data so that it is all saved arbitrarily many times at different timesteps - when you index the data at a given step, each value is defined as its most recent saved state
// TODO: add a save method that saves only elements from a given index
// TODO: add pairwise calculations (contacts), add hessian, add stress, add pressure
// TODO: add relevant plotting to python library - plot, animate, draw contact network, color by array
// TODO: add box resizing
// TODO: add box resizing - needs to sync with cell sizes and resize particle positions
// TODO: remove redundancy in the packing fraction box resizing calculations
// TODO: add overrides to system level operations for uniform operation (i.e. passing a single scalar dt instead of an array)
// TODO: implement cell list for rigid bumpy particles
// TODO: implement all remaining unimplemented functions in rigid bumpy
// TODO: add remaining integrators from dpcuda2
// TODO: remove vertex mass from rigid bumpy
// TODO: remove redundancies in the jamming and minimizing code - try to merge the different functions
// TODO: improve the set/restore last state code
// TODO: verify that the set/restore last state code works with cell lists
// TODO: verify that the box resizing code works with cell lists
// TODO: test various position, velocity, force update schemes for the rigid bumpy particles
// TODO: add post-processing routines
// TODO: add particle init function (performs tested series of steps to set up all data)
// TODO: add a particle validate function (checks everything is synced, sizes match expectation, etc,)
// TODO: mirror particle classes in python as data classes
// TODO: refactor constant and namespace naming for better clarity
// TODO: refactor function naming for better clarity
// TODO: add size verification function - check that all n_particle length arrays are the same length and whatnot
// TODO: add more tests for poly particle - vertex PE sum should match particle PE
// TODO: elevate certain functions for rigid bumpy to poly particle - maybe make an intermediary - rigid poly?
// TODO: add fast data output system, works in parallel to main routine
// TODO: design a system for getting arbitrary data and running arbitrary functions from particle object
// TODO: implement tests (mainly for disk) with non-uniform values of radii and mass
// TODO: add more tests to base particle (ke and pe sums, n_vertices, n_particles, n_systems, etc) and base point particle

// TODO: split out tests and scripts

// TODO: add sample within polygon - (requires polygon vertices, offsets, particle id) - launch kernel over polygon particle id size, only sample particles with a polygon

// TODO: refactor the global namespace from geo to something more descriptive

// TODO: profile standard vs bucket cell building in the point particles
// TODO: implement various cell list schemes for rigid particles (vertex level vs particle level)
// TODO: add a rigid particle with core
// TODO: profile kernel size
// TODO: profile speed vs system size and number of particles - how to speed up? - what is ideal dimension?

// TODO: build system-concatenated data from separate data inputs

// TODO: atomic restart

// TODO: when setting random positions, may want to add a branch for updating displacements and cell-list rebuild flag or at least enforce a neighbor rebuild?

// Later:
// TODO: implement a morton or hilbert based cell list stencil // TODO: raise certain common methods to higher-level classes (some components of the neighbor list)
// TODO: overhaul job manager
// - non-blocking job db - cannot crash if mulitple accesses occur
// - break jobs into blocks (initialization, program run, post process)
// - if a block fails, resume it

// TODO: implement all prior scripts