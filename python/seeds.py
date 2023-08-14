import os
import pyopencl as cl
import numpy as np


class Seeds:

    def __init__(self, N_threads, wdir):

        print("Generating seeds... ", end="")

        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)

        with open(os.path.join(wdir, "../opencl", "seed.cl"), 'r') as file:
            program_source = file.read()

        include_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../opencl")
        build_options = ['-I', include_path]

        program = cl.Program(context, program_source).build(options=build_options)

        seed = np.random.randint(0, 2**64, size=N_threads, dtype=np.uint64)
        seed_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=seed)

        # Define the struct-like Photon class

        states_type = np.dtype([
            ('a', np.uint32),
            ('b', np.uint32),
            ("c", np.uint32),
            ("d", np.uint32)
            ])

        self.states = np.zeros(N_threads, dtype=states_type)
        states_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, self.states.nbytes)

        kernel = program.seeds_generator
        kernel.set_args(seed_buffer, states_buffer)

        gpu_sim = cl.enqueue_nd_range_kernel(queue, kernel, (N_threads,), None)
        gpu_sim.wait()

        # Read the result back to the host

        cl.enqueue_copy(queue, self.states, states_buffer)

        print("Done!")
