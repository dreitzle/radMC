#ifndef SCATTERING_H
#define SCATTERING_H

#include <MC_planar_header.cl>
#include <tyche_i.cl>

#if !defined(C_GF)

float phase_function(const float cost)
{
    return M_1_4PI_F;
}

#else // g != 0

float phase_function(const float cost)
{
    return DIVIDE_C(M_1_4PI_F*(1.0f-C_GF*C_GF),POWR_C(1.0f+C_GF*C_GF-2.0f*C_GF*cost,1.5f));
}

#endif //!defined(C_GF)

void scatter_photon(Dir3d* direction, tyche_i_state* state)
{
    #if !defined(C_GF) // g = 0
    
    direction->cost = clamp(tyche_i_float((*state))*2.0f-1.0f,-1.0f,1.0f);
    direction->sint = SQRT_C(1.0f-direction->cost*direction->cost);
    direction->sinp = sincos(tyche_i_float((*state))*M_2PI_F, &(direction->cosp));
    
    #else // g != 0

    // sample angles according to Henyey-Greenstein phase function
    float temp = DIVIDE_C((1.0f - C_GF*C_GF),(1.0f - C_GF + 2.0f*C_GF*tyche_i_float((*state))));
    float cost = DIVIDE_C((1.0f + C_GF*C_GF - temp*temp),(2.0f*C_GF));

    cost = clamp(cost,-1.0f,1.0f);
    
    float sint = SQRT_C(1.0f-cost*cost);

    float cosp;
    float sinp = sincos(tyche_i_float((*state))*M_2PI_F, &cosp);

    float x = clamp(-sinp*sint*direction->sinp+direction->cosp*(cost*direction->sint+cosp*sint*direction->cost),-1.0f,1.0f);
    float y = clamp(cost*direction->sinp*direction->sint+sint*(direction->cosp*sinp+cosp*direction->cost*direction->sinp),-1.0f,1.0f);
    float z = clamp(direction->cost*cost-cosp*sint*direction->sint,-1.0f,1.0f);
    
    float den = SQRT_C(x*x+y*y);
    
    if(den < 1e-6f)
    {
        direction->cost = sign(z);
        direction->sint = 0.0f;
        direction->cosp = 1.0f;
        direction->sinp = 0.0f;
    }
    else
    {
        direction->cost = z;
        direction->sint = SQRT_C(1.0f-z*z);
        direction->cosp = DIVIDE_C(x,den);
        direction->sinp = DIVIDE_C(y,den);
    }
    
    #endif //!defined(C_GF)
}

#endif //SCATTERING_H
