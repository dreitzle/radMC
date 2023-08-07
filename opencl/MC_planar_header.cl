#ifndef MC_PLANAR_H
#define MC_PLANAR_H

#define M_2PI_F 6.2831853071795864769f
#define M_1_4PI_F 0.07957747154594767f

typedef struct {
    float cost;
    float sint;
    float cosp;
    float sinp;
} Dir3d;

typedef struct {
    Dir3d dir;
    float zpos;
    float weight;
    uint scat_counter;
} Photon;

#endif
