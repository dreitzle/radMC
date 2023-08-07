#include "tyche_i.cl"

__kernel void seeds_generator(__global unsigned long* seed, __global tyche_i_state* output_state)
{
    int thread_id = get_global_id(0);

    tyche_i_state state;
    tyche_i_seed(&state, seed[thread_id]);

    output_state[thread_id] = state;
}
