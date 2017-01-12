/*
 * Copyright 2016 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Integer implementation of the Discrete Cosine Transform (DCT)
//
// Note! DCT output is kept scaled by 16, to retain maximum 16bit precision

#include "guetzli/fdct.h"

namespace guetzli {

namespace {

///////////////////////////////////////////////////////////////////////////////
// Cosine table: C(k) = cos(k.pi/16)/sqrt(2), k = 1..7 using 15 bits signed
const coeff_t kTable04[7] = { 22725, 21407, 19266, 16384, 12873,  8867, 4520 };
// rows #1 and #7 are pre-multiplied by 2.C(1) before the 2nd pass.
// This multiply is merged in the table of constants used during 1st pass:
const coeff_t kTable17[7] = { 31521, 29692, 26722, 22725, 17855, 12299, 6270 };
// rows #2 and #6 are pre-multiplied by 2.C(2):
const coeff_t kTable26[7] = { 29692, 27969, 25172, 21407, 16819, 11585, 5906 };
// rows #3 and #5 are pre-multiplied by 2.C(3):
const coeff_t kTable35[7] = { 26722, 25172, 22654, 19266, 15137, 10426, 5315 };

///////////////////////////////////////////////////////////////////////////////
// Constants (15bit precision) and C macros for IDCT vertical pass

#define kTan1   (13036)   // = tan(pi/16)
#define kTan2   (27146)   // = tan(2.pi/16) = sqrt(2) - 1.
#define kTan3m1 (-21746)  // = tan(3.pi/16) - 1
#define k2Sqrt2 (23170)   // = 1 / 2.sqrt(2)

  // performs: {a,b} <- {a-b, a+b}, without saturation
#define BUTTERFLY(a, b) do {   \
  SUB((a), (b));               \
  ADD((b), (b));               \
  ADD((b), (a));               \
} while (0)

///////////////////////////////////////////////////////////////////////////////
// Constants for DCT horizontal pass

// Note about the CORRECT_LSB macro:
// using 16bit fixed-point constants, we often compute products like:
// p = (A*x + B*y + 32768) >> 16 by adding two sub-terms q = (A*x) >> 16
// and r = (B*y) >> 16 together. Statistically, we have p = q + r + 1
// in 3/4 of the cases. This can be easily seen from the relation:
//   (a + b + 1) >> 1 = (a >> 1) + (b >> 1) + ((a|b)&1)
// The approximation we are doing is replacing ((a|b)&1) by 1.
// In practice, this is a slightly more involved because the constants A and B
// have also been rounded compared to their exact floating point value.
// However, all in all the correction is quite small, and CORRECT_LSB can
// be defined empty if needed.

#define COLUMN_DCT8(in) do { \
  LOAD(m0, (in)[0 * 8]);     \
  LOAD(m2, (in)[2 * 8]);     \
  LOAD(m7, (in)[7 * 8]);     \
  LOAD(m5, (in)[5 * 8]);     \
                             \
  BUTTERFLY(m0, m7);         \
  BUTTERFLY(m2, m5);         \
                             \
  LOAD(m3, (in)[3 * 8]);     \
  LOAD(m4, (in)[4 * 8]);     \
  BUTTERFLY(m3, m4);         \
                             \
  LOAD(m6, (in)[6 * 8]);     \
  LOAD(m1, (in)[1 * 8]);     \
  BUTTERFLY(m1, m6);         \
  BUTTERFLY(m7, m4);         \
  BUTTERFLY(m6, m5);         \
                             \
  /* RowIdct() needs 15bits fixed-point input, when the output from   */ \
  /* ColumnIdct() would be 12bits. We are better doing the shift by 3 */ \
  /* now instead of in RowIdct(), because we have some multiplies to  */ \
  /* perform, that can take advantage of the extra 3bits precision.   */ \
  LSHIFT(m4, 3);             \
  LSHIFT(m5, 3);             \
  BUTTERFLY(m4, m5);         \
  STORE16((in)[0 * 8], m5);  \
  STORE16((in)[4 * 8], m4);  \
                             \
  LSHIFT(m7, 3);             \
  LSHIFT(m6, 3);             \
  LSHIFT(m3, 3);             \
  LSHIFT(m0, 3);             \
                             \
  LOAD_CST(m4, kTan2);       \
  m5 = m4;                   \
  MULT(m4, m7);              \
  MULT(m5, m6);              \
  SUB(m4, m6);               \
  ADD(m5, m7);               \
  STORE16((in)[2 * 8], m5);  \
  STORE16((in)[6 * 8], m4);  \
                             \
  /* We should be multiplying m6 by C4 = 1/sqrt(2) here, but we only have */ \
  /* the k2Sqrt2 = 1/(2.sqrt(2)) constant that fits into 15bits. So we    */ \
  /* shift by 4 instead of 3 to compensate for the additional 1/2 factor. */ \
  LOAD_CST(m6, k2Sqrt2);     \
  LSHIFT(m2, 3 + 1);         \
  LSHIFT(m1, 3 + 1);         \
  BUTTERFLY(m1, m2);         \
  MULT(m2, m6);              \
  MULT(m1, m6);              \
  BUTTERFLY(m3, m1);         \
  BUTTERFLY(m0, m2);         \
                             \
  LOAD_CST(m4, kTan3m1);     \
  LOAD_CST(m5, kTan1);       \
  m7 = m3;                   \
  m6 = m1;                   \
  MULT(m3, m4);              \
  MULT(m1, m5);              \
                             \
  ADD(m3, m7);               \
  ADD(m1, m2);               \
  CORRECT_LSB(m1);           \
  CORRECT_LSB(m3);           \
  MULT(m4, m0);              \
  MULT(m5, m2);              \
  ADD(m4, m0);               \
  SUB(m0, m3);               \
  ADD(m7, m4);               \
  SUB(m5, m6);               \
                             \
  STORE16((in)[1 * 8], m1);  \
  STORE16((in)[3 * 8], m0);  \
  STORE16((in)[5 * 8], m7);  \
  STORE16((in)[7 * 8], m5);  \
} while (0)


// these are the macro required by COLUMN_*
#define LOAD_CST(dst, src) (dst) = (src)
#define LOAD(dst, src) (dst) = (src)
#define MULT(a, b)  (a) = (((a) * (b)) >> 16)
#define ADD(a, b)   (a) = (a) + (b)
#define SUB(a, b)   (a) = (a) - (b)
#define LSHIFT(a, n) (a) = ((a) << (n))
#define STORE16(a, b) (a) = (b)
#define CORRECT_LSB(a) (a) += 1

// DCT vertical pass

inline void ColumnDct(coeff_t* in) {
  for (int i = 0; i < 8; ++i) {
    int m0, m1, m2, m3, m4, m5, m6, m7;
    COLUMN_DCT8(in + i);
  }
}

// DCT horizontal pass

// We don't really need to round before descaling, since we
// still have 4 bits of precision left as final scaled output.
#define DESCALE(a)  static_cast<coeff_t>((a) >> 16)

void RowDct(coeff_t* in, const coeff_t* table) {
  // The Fourier transform is an unitary operator, so we're basically
  // doing the transpose of RowIdct()
  const int a0 = in[0] + in[7];
  const int b0 = in[0] - in[7];
  const int a1 = in[1] + in[6];
  const int b1 = in[1] - in[6];
  const int a2 = in[2] + in[5];
  const int b2 = in[2] - in[5];
  const int a3 = in[3] + in[4];
  const int b3 = in[3] - in[4];

  // even part
  const int C2 = table[1];
  const int C4 = table[3];
  const int C6 = table[5];
  const int c0 = a0 + a3;
  const int c1 = a0 - a3;
  const int c2 = a1 + a2;
  const int c3 = a1 - a2;

  in[0] = DESCALE(C4 * (c0 + c2));
  in[4] = DESCALE(C4 * (c0 - c2));
  in[2] = DESCALE(C2 * c1 + C6 * c3);
  in[6] = DESCALE(C6 * c1 - C2 * c3);

  // odd part
  const int C1 = table[0];
  const int C3 = table[2];
  const int C5 = table[4];
  const int C7 = table[6];
  in[1] = DESCALE(C1 * b0 + C3 * b1 + C5 * b2 + C7 * b3);
  in[3] = DESCALE(C3 * b0 - C7 * b1 - C1 * b2 - C5 * b3);
  in[5] = DESCALE(C5 * b0 - C1 * b1 + C7 * b2 + C3 * b3);
  in[7] = DESCALE(C7 * b0 - C5 * b1 + C3 * b2 - C1 * b3);
}
#undef DESCALE
#undef LOAD_CST
#undef LOAD
#undef MULT
#undef ADD
#undef SUB
#undef LSHIFT
#undef STORE16
#undef CORRECT_LSB
#undef kTan1
#undef kTan2
#undef kTan3m1
#undef k2Sqrt2
#undef BUTTERFLY
#undef COLUMN_DCT8

}  // namespace

///////////////////////////////////////////////////////////////////////////////
// visible FDCT callable functions

void ComputeBlockDCT(coeff_t* coeffs) {
  ColumnDct(coeffs);
  RowDct(coeffs + 0 * 8, kTable04);
  RowDct(coeffs + 1 * 8, kTable17);
  RowDct(coeffs + 2 * 8, kTable26);
  RowDct(coeffs + 3 * 8, kTable35);
  RowDct(coeffs + 4 * 8, kTable04);
  RowDct(coeffs + 5 * 8, kTable35);
  RowDct(coeffs + 6 * 8, kTable26);
  RowDct(coeffs + 7 * 8, kTable17);
}

}  // namespace guetzli
