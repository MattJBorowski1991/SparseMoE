Profiling analysis of swizzle_ldmatrix. Motivation for performing swizzling via ldmatrix and mma.sync primitives: 

swizzle_xor swizzles the stored layout but still pays extra reorder/unswizzle cost, while swizzle_ldmatrix tries to make the swizzled layout match the tensor-core consumer path directly.

So the motivation is:

reduce shared-memory bank conflicts at the point that matters most
avoid extra shared-memory reshuffling
feed ldmatrix/mma.sync in a layout they can consume directly

PTX is not inherently required here, what is required is that the consumer can read the swizzled layout directly. PTX-level ldmatrix/mma.sync just gives you finer control to do that.


Why isn't unswizzle required for thie ldmatrix path , but is required for the XOR path: 

wmma::load_matrix_sync requires a linear row-major layout in shared memory. It reads elements at &tile[row*ld + col] and internally decides which thread gets which element according to a fixed hardware mapping. If you rearranged the data (swizzled), the hardware still reads at the same linear addresses, so it consumes the wrong values — hence you must unswizzle first to restore the expected linear order.

ldmatrix is different in a key way: you control the address each thread supplies. Each of the 32 threads in the warp provides its own pointer into shared memory, and the hardware gathers 8 bytes from each of those 8-byte-aligned addresses and distributes them into registers according to the mma.sync fragment layout. So:

You design your swizzle so that the address thread t supplies lands on a different bank than the addresses supplied by threads that are 8-thread groups away.
The hardware reads all 32 addresses simultaneously, each hitting a different bank — no conflict.
The gathered register values are already in the exact layout mma.sync expects, because you designed the swizzle to match mma.sync's register layout.
In other words:

wmma::load_matrix_sync — layout is hardware-fixed; you must give it linear data.
ldmatrix — layout is pointer-driven; the swizzled pointers are the consumption — there is no "before" and "after", the hardware gathers directly from wherever you point it.
That is exactly why swizzle_ldmatrix removes the unswizzle overhead: the swizzled write addresses and the PTX-driven read addresses are co-designed so the data lands precisely where the hardware needs it to be.