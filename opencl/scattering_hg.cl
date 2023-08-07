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
    return M_1_4PI_F*(1.0f-C_GF*C_GF)/pow(1.0f+C_GF*C_GF-2.0f*C_GF*cost,1.5f);
}

#endif //!defined(C_GF)

void scatter_photon(Dir3d* direction, tyche_i_state* state)
{
    #if !defined(C_GF) // g = 0
    
    direction->cost = tyche_i_float((*state))*2.0f-1.0f;
    direction->sint = sqrt(1.0f-direction->cost*direction->cost);
    direction->sinp = sincos(tyche_i_float((*state))*M_2PI_F, &(direction->cosp));
    
    #else // g != 0

    // sample angles according to Henyey-Greenstein phase function
    float temp = (1.0f - C_GF*C_GF)/(1.0f - C_GF + 2.0f*C_GF*tyche_i_float((*state)));
    float cost = (1.0f + C_GF*C_GF - temp*temp)/(2.0f*C_GF);

    cost = clamp(cost,-1.0f,1.0f);
    
    float sint = sqrt(1.0f-cost*cost);

    float cosp;
    float sinp = sincos(tyche_i_float((*state))*M_2PI_F, &cosp);

    float x = -sinp*sint*direction->sinp+direction->cosp*(cost*direction->sint+cosp*sint*direction->cost);
    float y = cost*direction->sinp*direction->sint+sint*(direction->cosp*sinp+cosp*direction->cost*direction->sinp);
    float z = direction->cost*cost-cosp*sint*direction->sint;

    x = clamp(x,-1.0f,1.0f);
    y = clamp(y,-1.0f,1.0f);
    z = clamp(z,-1.0f,1.0f);
    
    float den = sqrt(x*x+y*y);
    
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
        direction->sint = sqrt(1.0f-z*z);
        direction->cosp = x/den;
        direction->sinp = y/den;
    }
    
    #endif //!defined(C_GF)
}

#endif //SCATTERING_H