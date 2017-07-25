#pragma once

#ifdef __USE_OPENCL__

template<typename T>
inline void clSetKernelArgK(cl_kernel k, int idx, T* t)
{
    clSetKernelArg(k, idx, sizeof(T), t);
}

template<>
inline void clSetKernelArgK(cl_kernel k, int idx, int* t)
{
    cl_int c = *t;
    clSetKernelArg(k, idx, sizeof(cl_int), &c);
}

template<>
inline void clSetKernelArgK(cl_kernel k, int idx, const int* t)
{
    cl_int c = *t;
    clSetKernelArg(k, idx, sizeof(cl_int), &c);
}

template<>
inline void clSetKernelArgK(cl_kernel k, int idx, size_t* t)
{
    cl_int c = *t;
    clSetKernelArg(k, idx, sizeof(cl_int), &c);
}

template<>
inline void clSetKernelArgK(cl_kernel k, int idx, const size_t* t)
{
    cl_int c = *t;
    clSetKernelArg(k, idx, sizeof(cl_int), &c);
}

template<typename T0>
inline void clSetKernelArgEx(cl_kernel k, T0* t0)
{
    clSetKernelArgK(k, 0, t0);
}

template<typename T0, typename T1>
inline void clSetKernelArgEx(cl_kernel k, T0* t0, T1* t1)
{
    clSetKernelArgK(k, 1, t1);
    clSetKernelArgEx(k, t0);
}

template<typename T0, typename T1, typename T2>
inline void clSetKernelArgEx(cl_kernel k, T0* t0, T1* t1, T2* t2)
{
    clSetKernelArgK(k, 2, t2);
    clSetKernelArgEx(k, t0, t1);
}

template<typename T0, typename T1, typename T2, typename T3>
inline void clSetKernelArgEx(cl_kernel k, T0* t0, T1* t1, T2* t2, T3* t3)
{
    clSetKernelArgK(k, 3, t3);
    clSetKernelArgEx(k, t0, t1, t2);
}

template<typename T0, typename T1, typename T2, typename T3, typename T4>
inline void clSetKernelArgEx(cl_kernel k, T0* t0, T1* t1, T2* t2, T3* t3, T4* t4)
{
    clSetKernelArgK(k, 4, t4);
    clSetKernelArgEx(k, t0, t1, t2, t3);
}

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
inline void clSetKernelArgEx(cl_kernel k, T0* t0, T1* t1, T2* t2, T3* t3, T4* t4, T5* t5)
{
    clSetKernelArgK(k, 5, t5);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4);
}

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
inline void clSetKernelArgEx(cl_kernel k, T0* t0, T1* t1, T2* t2, T3* t3, T4* t4, T5* t5, T6* t6)
{
    clSetKernelArgK(k, 6, t6);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5);
}

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
inline void clSetKernelArgEx(cl_kernel k, T0* t0, T1* t1, T2* t2, T3* t3, T4* t4, T5* t5, T6* t6, T7* t7)
{
    clSetKernelArgK(k, 7, t7);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6);
}

template<typename T0, typename T1, typename T2, typename T3, typename T4, 
         typename T5, typename T6, typename T7, typename T8>
inline void clSetKernelArgEx(cl_kernel k, T0* t0, T1* t1, T2* t2, T3* t3, T4* t4, T5* t5, T6* t6, T7* t7, T8* t8)
{
    clSetKernelArgK(k, 8, t8);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7);
}

template<typename T0, typename T1, typename T2, typename T3, typename T4,
         typename T5, typename T6, typename T7, typename T8, typename T9>
inline void clSetKernelArgEx(cl_kernel k, T0* t0, T1* t1, T2* t2, T3* t3, T4* t4, T5* t5, T6* t6, T7* t7, T8* t8, T9* t9)
{
    clSetKernelArgK(k, 9, t9);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8);
}

template<typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10>
    inline void clSetKernelArgEx(cl_kernel k, T0* t0, T1* t1, T2* t2, T3* t3, T4* t4, T5* t5, T6* t6, T7* t7, T8* t8, T9* t9, T10* t10)
{
    clSetKernelArgK(k, 10, t10);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9);
}

template<typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10, typename T11>
    inline void clSetKernelArgEx(cl_kernel k, T0* t0, T1* t1, T2* t2, T3* t3, T4* t4, T5* t5, T6* t6, T7* t7, T8* t8, T9* t9, T10* t10, T11* t11)
{
    clSetKernelArgK(k, 11, t11);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10);
}

template<typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10, typename T11, typename T12>
    inline void clSetKernelArgEx(cl_kernel k, T0* t0, T1* t1, T2* t2, T3* t3, T4* t4, 
          T5* t5, T6* t6, T7* t7, T8* t8, T9* t9, 
          T10* t10, T11* t11, T12* t12)
{
    clSetKernelArgK(k, 12, t12);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11);
}

template<typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10, typename T11, typename T12, typename T13>
    inline void clSetKernelArgEx(cl_kernel k, T0* t0, T1* t1, T2* t2, T3* t3, T4* t4,
        T5* t5, T6* t6, T7* t7, T8* t8, T9* t9,
        T10* t10, T11* t11, T12* t12, T13* t13)
{
    clSetKernelArgK(k, 13, t13);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12);
}

template<typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10, typename T11, typename T12, typename T13, typename T14>
    inline void clSetKernelArgEx(cl_kernel k, T0* t0, T1* t1, T2* t2, T3* t3, T4* t4,
        T5* t5, T6* t6, T7* t7, T8* t8, T9* t9,
        T10* t10, T11* t11, T12* t12, T13* t13,
        T14* t14)
{
    clSetKernelArgK(k, 14, t14);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13);
}

template<
    typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10, typename T11, typename T12, typename T13, typename T14,
    typename T15>
    inline void clSetKernelArgEx(cl_kernel k,
        T0* t0, T1* t1, T2* t2, T3* t3, T4* t4,
        T5* t5, T6* t6, T7* t7, T8* t8, T9* t9,
        T10* t10, T11* t11, T12* t12, T13* t13, T14* t14,
        T15* t15)
{
    clSetKernelArgK(k, 15, t15);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14);
}

template<
    typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10, typename T11, typename T12, typename T13, typename T14,
    typename T15, typename T16>
    inline void clSetKernelArgEx(cl_kernel k,
        T0* t0, T1* t1, T2* t2, T3* t3, T4* t4,
        T5* t5, T6* t6, T7* t7, T8* t8, T9* t9,
        T10* t10, T11* t11, T12* t12, T13* t13, T14* t14,
        T15* t15, T16* t16)
{
    clSetKernelArgK(k, 16, t16);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15);
}

template<
    typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10, typename T11, typename T12, typename T13, typename T14,
    typename T15, typename T16, typename T17>
    inline void clSetKernelArgEx(cl_kernel k,
        T0* t0, T1* t1, T2* t2, T3* t3, T4* t4,
        T5* t5, T6* t6, T7* t7, T8* t8, T9* t9,
        T10* t10, T11* t11, T12* t12, T13* t13, T14* t14,
        T15* t15, T16* t16, T17* t17)
{
    clSetKernelArgK(k, 17, t17);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16);
}

template<
    typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10, typename T11, typename T12, typename T13, typename T14,
    typename T15, typename T16, typename T17, typename T18>
    inline void clSetKernelArgEx(cl_kernel k,
        T0* t0, T1* t1, T2* t2, T3* t3, T4* t4,
        T5* t5, T6* t6, T7* t7, T8* t8, T9* t9,
        T10* t10, T11* t11, T12* t12, T13* t13, T14* t14,
        T15* t15, T16* t16, T17* t17, T18* t18)
{
    clSetKernelArgK(k, 18, t18);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17);
}

template<
    typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10, typename T11, typename T12, typename T13, typename T14,
    typename T15, typename T16, typename T17, typename T18, typename T19>
    inline void clSetKernelArgEx(cl_kernel k,
        T0* t0, T1* t1, T2* t2, T3* t3, T4* t4,
        T5* t5, T6* t6, T7* t7, T8* t8, T9* t9,
        T10* t10, T11* t11, T12* t12, T13* t13, T14* t14,
        T15* t15, T16* t16, T17* t17, T18* t18, T19* t19)
{
    clSetKernelArgK(k, 19, t19);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18);
}

template<
    typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10, typename T11, typename T12, typename T13, typename T14,
    typename T15, typename T16, typename T17, typename T18, typename T19,
    typename T20>
    inline void clSetKernelArgEx(cl_kernel k,
        T0* t0, T1* t1, T2* t2, T3* t3, T4* t4,
        T5* t5, T6* t6, T7* t7, T8* t8, T9* t9,
        T10* t10, T11* t11, T12* t12, T13* t13, T14* t14,
        T15* t15, T16* t16, T17* t17, T18* t18, T19* t19,
        T20* t20)
{
    clSetKernelArgK(k, 20, t20);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19);
}

template<
    typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10, typename T11, typename T12, typename T13, typename T14,
    typename T15, typename T16, typename T17, typename T18, typename T19,
    typename T20, typename T21>
    inline void clSetKernelArgEx(cl_kernel k,
        T0* t0, T1* t1, T2* t2, T3* t3, T4* t4,
        T5* t5, T6* t6, T7* t7, T8* t8, T9* t9,
        T10* t10, T11* t11, T12* t12, T13* t13, T14* t14,
        T15* t15, T16* t16, T17* t17, T18* t18, T19* t19,
        T20* t20, T21* t21)
{
    clSetKernelArgK(k, 21, t21);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20);
}

template<
    typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10, typename T11, typename T12, typename T13, typename T14,
    typename T15, typename T16, typename T17, typename T18, typename T19,
    typename T20, typename T21, typename T22>
    inline void clSetKernelArgEx(cl_kernel k,
        T0* t0, T1* t1, T2* t2, T3* t3, T4* t4,
        T5* t5, T6* t6, T7* t7, T8* t8, T9* t9,
        T10* t10, T11* t11, T12* t12, T13* t13, T14* t14,
        T15* t15, T16* t16, T17* t17, T18* t18, T19* t19,
        T20* t20, T21* t21, T22* t22)
{
    clSetKernelArgK(k, 22, t22);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21);
}

template<
    typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10, typename T11, typename T12, typename T13, typename T14,
    typename T15, typename T16, typename T17, typename T18, typename T19,
    typename T20, typename T21, typename T22, typename T23>
    inline void clSetKernelArgEx(cl_kernel k,
        T0* t0, T1* t1, T2* t2, T3* t3, T4* t4,
        T5* t5, T6* t6, T7* t7, T8* t8, T9* t9,
        T10* t10, T11* t11, T12* t12, T13* t13, T14* t14,
        T15* t15, T16* t16, T17* t17, T18* t18, T19* t19,
        T20* t20, T21* t21, T22* t22, T23* t23)
{
    clSetKernelArgK(k, 23, t23);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22);
}

template<
    typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename T6, typename T7, typename T8, typename T9,
    typename T10, typename T11, typename T12, typename T13, typename T14,
    typename T15, typename T16, typename T17, typename T18, typename T19,
    typename T20, typename T21, typename T22, typename T23, typename T24>
inline void clSetKernelArgEx(cl_kernel k, 
        T0* t0, T1* t1, T2* t2, T3* t3, T4* t4,
        T5* t5, T6* t6, T7* t7, T8* t8, T9* t9,
        T10* t10, T11* t11, T12* t12, T13* t13, T14* t14, 
        T15* t15, T16* t16, T17* t17, T18* t18, T19* t19,
        T20* t20, T21* t21, T22* t22, T23* t23, T24* t24)
{
    clSetKernelArgK(k, 24, t24);
    clSetKernelArgEx(k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23);
}

#endif // __USE_OPENCL__