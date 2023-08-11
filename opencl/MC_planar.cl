#define NATIVE_MATH
#include <native_math.cl>

#include <MC_planar_header.cl>
#include <tyche_i.cl>
#include <scattering_hg.cl>

#ifdef USE_LF
    #include <detector_lf.cl>
#else
    #include <detector_integral.cl>
#endif

void reset_photon(Photon* photon)
{
    // start new photon: pencil beam perpendicular to surface
    photon->zpos = 0.0f;
    photon->dir.cost = 1.0f;
    photon->dir.sint = 0.0f;
    photon->dir.cosp = 1.0f;
    photon->dir.sinp = 0.0f;
    photon->weight = 1.0f;
    photon->scat_counter = 0u;
}

__kernel void init_photons(__global Photon* photons_d)
{
    // get thread id and total number of threads
    int thread_id = get_global_id(0);

    // init photon
    Photon photon;
    reset_photon(&photon);
    photons_d[thread_id] = photon;
}

__kernel void run(__global tyche_i_state* rng_states_d,
                  __global unsigned long* simulated_photons_pthread_d, 
                  __global Photon* photons_d, 
                  __global float* detector_d,
                  __constant float* cost_points,
                  __constant float* phi_points)
{
    // get thread id and total number of threads
    int thread_id = get_global_id(0);
    
    // initializing RNG "tyche_i_float(state)"
    tyche_i_state state = rng_states_d[thread_id];

    // initializing photon struct 
    Photon photon = photons_d[thread_id];

    // initialize local detector
    float detector_loc[N_COSTHETA*N_PHI];
    
    for(int idx = 0; idx < N_COSTHETA*N_PHI; ++idx)
        detector_loc[idx] = 0.0f;
    
    bool need_reset = false;
    
    // Main photon loop
    for(uint n_scat_pthread = 0; n_scat_pthread < N_SCAT_PTHREAD; ++n_scat_pthread)
    {
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 

        // sample free path length
        float free_path_length = DIVIDE_C(-LOG_C(tyche_i_float(state)),C_MUS); 
        
        // save old z position for radiance calculation
        float zpos_start = photon.zpos;
        
        // calc new z position
        photon.zpos = MAD_C(free_path_length,photon.dir.cost,photon.zpos);
        
        if (photon.zpos <= 0.0f)
        {
            // distance to surface along photon direction
            free_path_length = DIVIDE_C(-zpos_start,photon.dir.cost);

            // exit point of the photon
            photon.zpos = 0.0f;

            // reset later, call calc_rad_contribution only once
            need_reset = true;
        }

        // calculate contribution to radiance
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
        calc_rad_contribution(&photon, free_path_length, zpos_start, detector_loc, cost_points, phi_points);

        // reset or scatter
        if(need_reset)
        {
            reset_photon(&photon);
            ++simulated_photons_pthread_d[thread_id];
            need_reset = false;
        }
        else
        {
            // Attenuate photon
            photon.weight *= EXP_C(-C_MUA*free_path_length);
    
            // scattering
            scatter_photon(&photon.dir, &state); // HG phase function
            ++photon.scat_counter;

            // end photon if max number of scatterings exceeded or weight below threshold
            if(photon.scat_counter >= N_SCAT_MAX || photon.weight < 1e-8f) 
            {
                reset_photon(&photon);
                ++simulated_photons_pthread_d[thread_id];
            }
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // save photon and rng_states for next run
    rng_states_d[thread_id] = state;
    photons_d[thread_id] = photon;

    // save local detector to global memory
    for(int idx = 0; idx < N_PHI*N_COSTHETA; ++idx)
        detector_d[idx + thread_id*N_PHI*N_COSTHETA] += detector_loc[idx];

}

__kernel void finish(__global tyche_i_state* rng_states_d,
                     __global unsigned long* simulated_photons_pthread_d, 
                     __global Photon* photons_d, 
                     __global float* detector_d,
                     __constant float* cost_points,
                     __constant float* phi_points)
{
    // get thread id and total number of threads
    int thread_id = get_global_id(0);
    
    // initializing RNG "tyche_i_float(state)"
    tyche_i_state state = rng_states_d[thread_id];

    // initializing photon struct 
    Photon photon = photons_d[thread_id];

    // initialize local detector
    float detector_loc[N_COSTHETA*N_PHI];
    
    for(int idx = 0; idx < N_COSTHETA*N_PHI; ++idx)
        detector_loc[idx] = 0.0f;
    
    bool done = false;
    
    // Main photon loop
    while(true)
    {
        // sample free path length
        float free_path_length = DIVIDE_C(-LOG_C(tyche_i_float(state)),C_MUS); 
        
        // save old z position for radiance calculation
        float zpos_start = photon.zpos;
        
        // calc new z position
        photon.zpos = MAD_C(free_path_length,photon.dir.cost,photon.zpos);
        
        if (photon.zpos <= 0.0f)
        {
            // distance to surface along photon direction
            free_path_length = DIVIDE_C(-zpos_start,photon.dir.cost);

            // exit point of the photon
            photon.zpos = 0.0f;
            
            done = true;
        }
        
        // calculate contribution to radiance
        calc_rad_contribution(&photon, free_path_length, zpos_start, detector_loc, cost_points, phi_points);
        
        if(done)
            break;

        // Attenuate photon
        photon.weight *= EXP_C(-C_MUA*free_path_length);

        // scattering
        scatter_photon(&photon.dir, &state); // HG phase function
        ++photon.scat_counter;

        // end photon if max number of scatterings exceeded or weight below threshold
        if(photon.scat_counter >= N_SCAT_MAX || photon.weight < 1e-8f) 
            break;
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    ++simulated_photons_pthread_d[thread_id];
    
    // save photon and rng_states for next run
    rng_states_d[thread_id] = state;
    photons_d[thread_id] = photon;

    // save local detector to global memory
    for(int idx = 0; idx < N_PHI*N_COSTHETA; ++idx)
       detector_d[idx + thread_id*N_PHI*N_COSTHETA] += detector_loc[idx];

}
