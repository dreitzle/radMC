#ifndef GPU_SIM_KERNEL_NATIVE_MATH_H
#define GPU_SIM_KERNEL_NATIVE_MATH_H

#ifdef NATIVE_MATH

    #define SINCOS_C(X, Y) ( (*(Y) = native_cos( (X) )), native_sin( (X) ) )
    #define SIN_C native_sin
    #define COS_C native_cos
    #define TAN_C native_tan
    #define EXP_C native_exp
    #define EXP2_C native_exp2
    #define EXP10_C native_exp10
    #define LOG_C native_log
    #define LOG2_C native_log2
    #define LOG10_C native_log10
    #define SQRT_C native_sqrt
    #define RSQRT_C native_rsqrt

    #define DIVIDE_C native_divide
    #define RECIP_C native_recip
    #define MAD_C mad
    #define POWR_C native_powr

#else

    #define SINCOS_C sincos
    #define SIN_C sin
    #define COS_C cos
    #define TAN_C tan
    #define EXP_C exp
    #define EXP2_C exp2
    #define EXP10_C exp10
    #define LOG_C log
    #define LOG2_C log2
    #define LOG10_C log10
    #define SQRT_C sqrt

    #ifdef __OPENCL_VERSION__
        #define RSQRT_C rsqrt
    #else
        #define RSQRT_C(X) ( 1.0f/sqrt(X) )
    #endif

    #define DIVIDE_C(X, Y) ( (X)/(Y) )
    #define RECIP_C(X) ( 1.0f/(X) )
    #define MAD_C(A, B, C) ( (A)*(B) + (C) )
    #define POWR_C powr

#endif

#endif /* GPU_SIM_KERNEL_NATIVE_MATH_H */

