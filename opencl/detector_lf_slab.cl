#ifndef DETECTOR_H
#define DETECTOR_H

#include <MC_planar_header.cl>
#include <tyche_i.cl>
#include <fresnel.cl>

void calc_rad_contribution(Photon* photon, const float l_path, const float z1,
                           float* detector_r_loc, float* detector_t_loc, __constant float* cost_points,
                           __constant float* phi_points)
{
    if(photon->zpos <= 0.0f || photon->zpos >= D_SLAB)
        return;

    // calc contribution to radiance
    const float cost = photon->dir.cost;
    const float sint = photon->dir.sint;
    const float cosp = photon->dir.cosp;
    const float sinp = photon->dir.sinp;
    
    // loop over cost values
    for(int cost_point_idx = 0; cost_point_idx < N_COSTHETA; ++cost_point_idx)
    {
        float cost_point = cost_points[cost_point_idx];
        float sint_point = SQRT_C(1.0f-cost_point*cost_point);

        // reflectivity for the considered cost
        float r0 = calc_reflectivity(cost_point);
        // attenuation along the reflection path
        float atten = EXP_C(DIVIDE_C(-C_MUT*D_SLAB,fabs(cost_point))); 
        
        // loop over phi values
        for(int phi_point_idx = 0; phi_point_idx < N_PHI; ++phi_point_idx)
        {
            float cosp_point;
            float sinp_point = sincos(phi_points[phi_point_idx],&cosp_point);
            
            float scal_prod1 = sint*sint_point*(cosp*cosp_point+sinp*sinp_point)+cost*cost_point;
            float scal_prod2 = sint*sint_point*(cosp*cosp_point+sinp*sinp_point)+cost*(-cost_point);
            
            float pf1 = phase_function(scal_prod1);
            float pf2 = phase_function(scal_prod2);

            float step_to_surface1 = -DIVIDE_C(photon->zpos,cost_point);
            float step_to_surface2 = DIVIDE_C(D_SLAB-photon->zpos, -cost_point);
            
            float att_sur1 = EXP_C(-C_MUT*step_to_surface1-C_MUA*l_path);
            float att_sur2 = EXP_C(-C_MUT*step_to_surface2-C_MUA*l_path);
            
            float photon_weight1 = photon->weight*pf1;
            float photon_weight2 = photon->weight*pf2;
            
            // from geometric series considerung all ballistic reflection paths
            float r_contribution1 = DIVIDE_C(1.0f, 1.0f - r0*r0*atten*atten);
            float r_contribution2 = DIVIDE_C(r0*atten, 1.0f - r0*r0*atten*atten);
            
           detector_r_loc[phi_point_idx + cost_point_idx*N_PHI] += DIVIDE_C(photon_weight1*r_contribution1*att_sur1+photon_weight2*r_contribution2*att_sur2,fabs(cost_point));
           detector_t_loc[phi_point_idx + cost_point_idx*N_PHI] += DIVIDE_C(photon_weight2*r_contribution1*att_sur2+photon_weight1*r_contribution2*att_sur1,fabs(cost_point)); 

        }
    }
}

#endif //DETECTOR_H
