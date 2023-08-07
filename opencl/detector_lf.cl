#ifndef DETECTOR_H
#define DETECTOR_H

#include <MC_planar_header.cl>
#include <tyche_i.cl>

void calc_rad_contribution(Photon* photon, const float l_path, const float z1,
                           float* detector_loc, __constant float* cost_points,
                           __constant float* phi_points)
{
    if(photon->zpos <= 0.0f)
        return;

    // calc contribution to radiance
    const float cost = photon->dir.cost;
    const float sint = photon->dir.sint;
    const float cosp = photon->dir.cosp;
    const float sinp = photon->dir.sinp;
    
    for(int cost_point_idx = 0; cost_point_idx < N_COSTHETA; ++cost_point_idx)
    {
        float cost_point = cost_points[cost_point_idx];
        float sint_point = sqrt(1.0f-cost_point*cost_point);

        for(int phi_point_idx = 0; phi_point_idx < N_PHI; ++phi_point_idx)
        {
            float cosp_point;
            float sinp_point = sincos(phi_points[phi_point_idx],&cosp_point);
            
            float scal_prod = sint*sint_point*(cosp*cosp_point+sinp*sinp_point)+cost*cost_point; 
            float pf = phase_function(scal_prod);

            float step_to_surface = -photon->zpos/cost_point;
            float photon_weight = photon->weight*exp(-C_MUT*step_to_surface-C_MUA*l_path);

           detector_loc[phi_point_idx + cost_point_idx*N_PHI] += pf*photon_weight/fabs(cost_point); 
        }
    }
}

#endif //DETECTOR_H