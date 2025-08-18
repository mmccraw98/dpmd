// TODO: finish box sampling protocol
// TODO: fix the system-sizing issue (for N total (across all systems) > 1e4, things start breaking)
// TODO: add box resizing - needs to sync with cell sizes and resize particle positions
// TODO: add fast data output
// TODO: design a system for getting arbitrary data and running arbitrary functions from particle object
// TODO: add size verification function - check that all n_particle length arrays are the same length and whatnot

// TODO: profile standard vs bucket cell building in the point particles
// TODO: profile kernel size
// TODO: profile speed vs system size and number of particles - how to speed up?  - what is ideal dimension?

// TODO: build system-concatenated data from separate data inputs

// Later:
// TODO: implement a morton or hilbert based cell list stencil
// TODO: overhaul job manager
//      - non-blocking job db - cannot crash if mulitple accesses occur
//      - break jobs into blocks (initialization, program run, post process)
//      - if a block fails, resume it
