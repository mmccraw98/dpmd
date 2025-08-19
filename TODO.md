// TODO: finish box sampling protocol
// TODO: add size verification function - check that all n_particle length arrays are the same length and whatnot
// TODO: fix the system-sizing issue (for N total (across all systems) > 1e4, things start breaking)
// TODO: add box resizing - needs to sync with cell sizes and resize particle positions
// TODO: add fast data output
// TODO: design a system for getting arbitrary data and running arbitrary functions from particle object
// TODO: implement tests (mainly for disk) with non-uniform values of radii and mass
// TODO: add more tests to base particle (ke and pe sums, n_vertices, n_particles, n_systems, etc) and base point particle

// TODO: refactor the global namespace from geo to something more descriptive

// TODO: profile standard vs bucket cell building in the point particles
// TODO: implement various cell list schemes for rigid particles (vertex level vs particle level)
// TODO: add a rigid particle with core
// TODO: profile kernel size
// TODO: profile speed vs system size and number of particles - how to speed up?  - what is ideal dimension?

// TODO: build system-concatenated data from separate data inputs

// TODO: atomic restart

// Later:
// TODO: implement a morton or hilbert based cell list stencil
// TODO: raise certain common methods to higher-level classes (some components of the neighbor list)
// TODO: overhaul job manager
//      - non-blocking job db - cannot crash if mulitple accesses occur
//      - break jobs into blocks (initialization, program run, post process)
//      - if a block fails, resume it

// TODO: implement all prior scripts