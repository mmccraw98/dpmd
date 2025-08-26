// TODO: test the wall forces and energy conservation
// TODO: test the jamming code using disks
// TODO: add tests for wall forces for disks
// TODO: add tests for rigid bumpy dynamics
// TODO: implement cell list for rigid bumpy particles
// TODO: add remaining integrators from dpcuda2
// TODO: remove vertex mass from rigid bumpy
// TODO: test various position, velocity, force update schemes for the rigid bumpy particles
// TODO: add particle init function (performs tested series of steps to set up all data)
// TODO: add a particle validate function (checks everything is synced, sizes match expectation, etc,)
// TODO: mirror particle classes in python as data classes
// TODO: refactor constant and namespace naming for better clarity
// TODO: refactor function naming for better clarity
// TODO: add size verification function - check that all n_particle length arrays are the same length and whatnot
// TODO: add more tests for poly particle - vertex PE sum should match particle PE
// TODO: elevate certain functions for rigid bumpy to poly particle - maybe make an intermediary - rigid poly?
// TODO: add box resizing - needs to sync with cell sizes and resize particle positions
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