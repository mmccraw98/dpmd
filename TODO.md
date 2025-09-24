// TODO: calculate system-average velocity
// TODO: add center-of-mass velocity cancelling

// TODO: implement system averaging (like system segmented sum but then divided by system size)

// TODO: implement stress - make sure the stress tensor for rigid bodies is used!  i am not sure it is right otherwise
// TODO: implement pressure

// TODO: implement hessian
// TODO: define units
// TODO: define proper energy scale for rigid bumpy particles
// TODO: if rigid bumpy particle with core requires multiple energy scales, implement them
// TODO: validate rigid bumpy particle with core

// TODO: add cell list to rigid bumpy particle


// TODO: fix the init writing (if input and output directories are the same, the init should NEVER be overwritten)

// TODO: simplify the output manager
// TODO: add support for different logging schemes (linear, logarithmic)
// TODO: add energy file
// TODO: output manager configs


// TODO: probably should NOT be writing anything in init that doesnt exist already, whether in append mode or not

// TODO: add a way to manually save an array to trajectory under the current step with optional array indices to save:

// TODO: redesign all data so that it is all saved arbitrarily many times at different timesteps - when you index the data at a given step, each value is defined as its most recent saved state
// TODO: add a save method that saves only elements from a given index
// TODO: add pairwise calculations (contacts), add hessian, add stress, add pressure
// TODO: add relevant plotting to python library - plot, animate, draw contact network
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
// TODO: profile speed vs system size and number of particles - how to speed up?  - what is ideal dimension?

// TODO: build system-concatenated data from separate data inputs

// TODO: atomic restart

// TODO: when setting random positions, may want to add a branch for updating displacements and cell-list rebuild flag or at least enforce a neighbor rebuild?

// Later:
// TODO: implement a morton or hilbert based cell list stencil
// TODO: raise certain common methods to higher-level classes (some components of the neighbor list)
// TODO: overhaul job manager
//      - non-blocking job db - cannot crash if mulitple accesses occur
//      - break jobs into blocks (initialization, program run, post process)
//      - if a block fails, resume it

// TODO: implement all prior scripts