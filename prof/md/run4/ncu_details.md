Profiling analysis of capacity_v3. Motivation for performing swizzling via ldmatrix and mma.sync primitives: 

capacity_v2 swizzles the stored layout but still pays extra reorder/unswizzle cost, while capacity_v3 tries to make the swizzled layout match the tensor-core consumer path directly.

So the motivation is:

reduce shared-memory bank conflicts at the point that matters most
avoid extra shared-memory reshuffling
feed ldmatrix/mma.sync in a layout they can consume directly

PTX is not inherently required here, what is required is that the consumer can read the swizzled layout directly. PTX-level ldmatrix/mma.sync just gives you finer control to do that.