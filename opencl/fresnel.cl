#ifndef FRESNEL_H
#define FRESNEL_H

float calc_reflectivity(const float cost)
{
    #if defined(C_N1) && defined(C_N2) // refractive index mismatch

    const float acost = clamp(fabs(cost),0.0f,1.0f);
    
    const float n_ratio = DIVIDE_C(C_N2,C_N1);
    const float cost_crit = n_ratio > 1.0f ? DIVIDE_C(SQRT_C(n_ratio*n_ratio-1.0f),n_ratio) : 0.0f; // cos(critical angle)
    
    if(acost > cost_crit) // incident angle is smaller than critical angle
    {
        if(acost < 0.999999f) // ray not perpendicular
        {
                const float cosb = SQRT_C(1.0f - n_ratio*n_ratio*(1.0f - cost*cost));

                const float rs = DIVIDE_C(acost - n_ratio*cosb,acost + n_ratio*cosb);
                const float rp = DIVIDE_C(cosb - n_ratio*acost,cosb + n_ratio*acost);
        
                return DIVIDE_C(rs*rs + rp*rp,2.0f);
        }
        else // perpendicular ray
        {
            const float rs = DIVIDE_C(C_N1-C_N2,C_N1+C_N2);
            
            return rs*rs;
        }
    }
    else // total reflection
    {
        return 1.0f;
    }
    
    #else 
    
    return 0.0f;
    
    #endif // defined(C_N1) && defined(C_N2)
}

#endif //FRESNEL_H
