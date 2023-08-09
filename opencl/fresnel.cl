#ifndef FRESNEL_H
#define FRESNEL_H

#include <tyche_i.cl>

float calc_reflectivity(const float cost)
{
    #if defined(C_N1) // refractive index mismatch
    
    float n_ratio = DIVIDE_C(C_N2,C_N1);
    float cost_crit = n_ratio > 1.0f ? DIVIDE_C(SQRT_C(n_ratio*n_ratio-1.0f),n_ratio) : 0.0f; // cos(critical angle)
    
    if(fabs(cost) > cost_crit) // incident angle is smaller than critical angle
    {
        if(fabs(cost) < 0.999999f) // ray not perpendicular
        {
        		float cosb = SQRT_C(1.0f - n_ratio*n_ratio*(1.0f - cost*cost));
        				
        		float rs = DIVIDE_C(fabs(cost) - n_ratio*cosb,fabs(cost) + n_ratio*cosb);
        		float rp = DIVIDE_C(cosb - n_ratio*fabs(cost),cosb + n_ratio*fabs(cost));
        
        		return DIVIDE_C(rs*rs + rp*rp,2.0f);
        }
        else // perpendicular ray
        {
            float rs = DIVIDE_C(C_N1-C_N2,C_N1+C_N2);
            
            return rs*rs;
        }
    }
    else // total reflection
    {
        return 1.0f;
    }
    
    #else 
    
    return 0.0f;
    
    #endif // defined(C_N1)
}

#endif //FRESNEL_H