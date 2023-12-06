#ifndef DETECTOR_H
#define DETECTOR_H

#include <MC_planar_header.cl>
#include <tyche_i.cl>

void calc_rad_contribution(Photon* photon, const float l_path, const float z1,
                           float* detector_loc, __constant float* cost_points,
                           __constant float* phi_points)
{
    // calc contribution to radiance
    const float cost = photon->dir.cost;
    const float sint = photon->dir.sint;
    const float cosp = photon->dir.cosp;
    const float sinp = photon->dir.sinp;
    
    for(int cost_point_idx = 0; cost_point_idx < N_COSTHETA; ++cost_point_idx)
    {
        float cost_point = cost_points[cost_point_idx];
        float sint_point;

        if(fabs(cost_point) < 1e-2f)
            sint_point = 1.0f-0.5f*cost_point*cost_point;
        else
            sint_point = sqrt(1.0f-cost_point*cost_point);

        for(int phi_point_idx = 0; phi_point_idx < N_PHI; ++phi_point_idx)
        {
            float cosp_point;
            float sinp_point = sincos(phi_points[phi_point_idx],&cosp_point);
            
            float scal_prod = sint*sint_point*(cosp*cosp_point+sinp*sinp_point)+cost*cost_point; 
            float pf = phase_function(scal_prod);

            float den = C_MUA - C_MUT*DIVIDE_C(cost,cost_point);

            float X;
            
            if(fabs(den) < 1e-4f)
            {
                //too small, use expansion
                float dexp = DIVIDE_C(EXP_C(DIVIDE_C(C_MUT,cost_point)*z1),cost_point);
                X = dexp*l_path*(1.0f-l_path*den*(0.5f-l_path*DIVIDE_C(den,6.0f)));
            }
            else
            {
                float dexp = EXP_C(DIVIDE_C(C_MUT,cost_point)*z1) - EXP_C(-C_MUA*l_path+DIVIDE_C(C_MUT,cost_point)*photon->zpos);
                X = DIVIDE_C(dexp,den*cost_point);
            }

            detector_loc[phi_point_idx + cost_point_idx*N_PHI] -= C_MUS*photon->weight*pf*X;
        }
    }
}

#endif //DETECTOR_H
