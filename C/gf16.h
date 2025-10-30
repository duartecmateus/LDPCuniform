#ifndef GF16_H 
#define GF16_H

#include <stdint.h> 
#include <stdbool.h>

//ifdef __cplusplus extern "C" { #endif

// Representation: 4-bit value in an unsigned char (0..15) 
typedef unsigned char gf16_t;

extern volatile int gf16__initialized;

// Polynomial: x^4 + x + 1 -> lower-bits mask 0x3 for reduction, top bit 0x8 
#define GF16_POLY_LO ((gf16_t)0x3) 
#define GF16_TOP_BIT ((gf16_t)0x8) 
#define GF16_MASK ((gf16_t)0xF)

// Public tables (filled by gf16_init). You can read them if needed.

 extern gf16_t gf16_mul_table[16][16]; 
 extern gf16_t gf16_inv_table[16];

// Initialize tables (idempotent). Must be called before using mult/inv unless // you rely on the lazy-init inside those functions. 
void gf16_init(void);

// Core operations 

gf16_t gf16_add(gf16_t a, gf16_t b); // addition/subtraction are the same (XOR) 

gf16_t gf16_sub(gf16_t a, gf16_t b);

// Multiplication via memorized 16x16 table static inline 
gf16_t gf16_mult(gf16_t a, gf16_t b);

// Inverse via memorized 16-entry table // Returns 0 for input 0 (no inverse). Use gf16_inv_checked for a safe check. 

gf16_t gf16_inv(gf16_t a);

// Checked inverse: returns false if a == 0 (no inverse).
bool gf16_inv_checked(gf16_t a, gf16_t* out);

// Optional short names as requested (add, sub, mult, inv). // Define GF16_SHORT_NAMES before including this header to expose them. 


gf16_t add(gf16_t a, gf16_t b);
gf16_t sub(gf16_t a, gf16_t b);
gf16_t mult(gf16_t a, gf16_t b); 
gf16_t inv(gf16_t a);


//#ifdef __cplusplus } #endif

#endif // GF16_H