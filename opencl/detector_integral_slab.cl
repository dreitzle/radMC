#ifndef DETECTOR_H
#define DETECTOR_H

#include <MC_planar_header.cl>
#include <tyche_i.cl>
#include <fresnel.cl>

void calc_rad_contribution(Photon* photon, const float l_path, const float z1,
                           float* detector_r_loc, float* detector_t_loc, __constant float* cost_points,
                           __constant float* phi_points)
{
    // calc contribution to radiance
    const float cost = photon->dir.cost;
    const float sint = photon->dir.sint;
    const float cosp = photon->dir.cosp;
    const float sinp = photon->dir.sinp;
    
    // loop over cost values
    for(int cost_point_idx = 0; cost_point_idx < N_COSTHETA; ++cost_point_idx)
    {
        float cost_point = cost_points[cost_point_idx];
        float sint_point;

        if(fabs(cost_point) < 1e-2f)
            sint_point = 1.0f-0.5f*cost_point*cost_point;
        else
            sint_point = sqrt(1.0f-cost_point*cost_point);

        // reflectivity for the considered cost
        float r0 = calc_reflectivity(cost_point);
        // attenuation along the reflection path
        float atten = EXP_C(DIVIDE_C(-C_MUT*D_SLAB,fabs(cost_point))); 
        float atten2 = EXP_C(DIVIDE_C(-2.0f*C_MUT*D_SLAB,fabs(cost_point))); 
        
        // loop over phi values
        for(int phi_point_idx = 0; phi_point_idx < N_PHI; ++phi_point_idx)
        {
            float cosp_point;
            float sinp_point = sincos(phi_points[phi_point_idx],&cosp_point);
            
            float scal_prod1 = sint*sint_point*(cosp*cosp_point+sinp*sinp_point)+cost*cost_point;
            float scal_prod2 = sint*sint_point*(cosp*cosp_point+sinp*sinp_point)+cost*(-cost_point);
            
            float pf1 = phase_function(scal_prod1);
            float pf2 = phase_function(scal_prod2);

            float den1 = C_MUA - C_MUT*DIVIDE_C(cost,cost_point);
            float den2 = C_MUA + C_MUT*DIVIDE_C(cost,cost_point);

            float X,Y;
            
            if(fabs(den1) < 1e-4f)
            {
                //too small, use expansion
                float tmp = l_path*(1.0f-l_path*den1*(0.5f-l_path*DIVIDE_C(den1,6.0f)));
                X = tmp*DIVIDE_C(EXP_C(DIVIDE_C(C_MUT,cost_point)*z1),cost_point);
            }
            else
                X = DIVIDE_C(EXP_C(DIVIDE_C(C_MUT,cost_point)*z1) - EXP_C(-C_MUA*l_path+DIVIDE_C(C_MUT,cost_point)*photon->zpos),den1*cost_point);

            if(fabs(den2) < 1e-4f)
            {
                //too small, use expansion
                float tmp = l_path*(1.0f-l_path*den2*(0.5f-l_path*DIVIDE_C(den2,6.0f)));
                Y = tmp*DIVIDE_C(EXP_C(DIVIDE_C(C_MUT,cost_point)*(D_SLAB-z1)),cost_point);
            }
            else
                Y = DIVIDE_C(EXP_C(DIVIDE_C(C_MUT,cost_point)*(D_SLAB-z1)) - EXP_C(-C_MUA*l_path+DIVIDE_C(C_MUT,cost_point)*(D_SLAB-photon->zpos)),den2*cost_point);

            float rfac = DIVIDE_C(r0,1.0f-r0*r0*atten2);
            float c1 = pf1*X + rfac*(r0*pf1*atten2*X+pf2*atten*Y);
            float c2 = pf2*Y + rfac*(r0*pf2*atten2*Y+pf1*atten*X);

           detector_r_loc[phi_point_idx + cost_point_idx*N_PHI] -= C_MUS*photon->weight*c1;
           detector_t_loc[phi_point_idx + cost_point_idx*N_PHI] -= C_MUS*photon->weight*c2;
        }
    }
}

#endif //DETECTOR_H
