#ifndef __CLGUETZLI_CL_H__
#define __CLGUETZLI_CL_H__

#ifdef __cplusplus
    #define __kernel
    #define __private
    #define __global
    #define __constant
    typedef unsigned char uchar;
    typedef unsigned short ushort;

    int get_global_id(int dim);
    int get_global_size(int dim);
    void set_global_id(int dim, int id);
    void set_global_size(int dim, int size);

    #ifdef __opencl
        typedef union ocl_channels_t
        {
            struct
            {
                float * r;
                float * g;
                float * b;
            };
            union
            {
                float *ch[3];
            };
        }ocl_channels;
    #else
        typedef union ocl_channels_t
        {
            struct
            {
                cl_mem r;
                cl_mem g;
                cl_mem b;
            };
            struct
            {
                cl_mem x;
                cl_mem y;
                cl_mem b;
            };
            union
            {
                cl_mem ch[3];
            };
        }ocl_channels;
    #endif
#else /*__cplusplus*/
    typedef union ocl_channels_t
    {
        struct
        {
            float * r;
            float * g;
            float * b;
        };

        union
        {
            float *ch[3];
        };
    }ocl_channels;

#endif /*__cplusplus*/

    typedef short coeff_t;

    typedef struct __channel_info_t
    {
        int factor;
        int block_width;
        int block_height;
        __global const coeff_t *coeff;
        __global const ushort  *pixel;
    }channel_info;

#endif /*__CLGUETZLI_CL_H__*/