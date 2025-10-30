#include "gf16.h"

gf16_t gf16_mul_table[16][16]; 
gf16_t gf16_inv_table[16];


volatile int gf16__initialized = 0;

// Multiply by x in GF(2^4) with modulus x^4 + x + 1 

static inline gf16_t gf16_xtime(gf16_t x) { 
	gf16_t r = (gf16_t)((x << 1) & GF16_MASK); 
	if (x & GF16_TOP_BIT) r ^= GF16_POLY_LO; 
	return r & GF16_MASK; 
}

gf16_t gf16_add(gf16_t a, gf16_t b) { return (a ^ b) & GF16_MASK;} // addition/subtraction are the same (XOR) 

gf16_t gf16_sub(gf16_t a, gf16_t b) { return gf16_add(a, b); }

// Multiplication via memorized 16x16 table static inline 
gf16_t gf16_mult(gf16_t a, gf16_t b) { // Lazy init in case gf16_init wasn't called extern volatile int gf16__initialized; 
	if (!gf16__initialized) gf16_init(); 
	return gf16_mul_table[a & GF16_MASK][b & GF16_MASK]; 
}

// Inverse via memorized 16-entry table // Returns 0 for input 0 (no inverse). Use gf16_inv_checked for a safe check. 

gf16_t gf16_inv(gf16_t a) { 
	extern volatile int gf16__initialized; 
	if (!gf16__initialized) gf16_init(); 
	return gf16_inv_table[a & GF16_MASK]; 
	}

// Checked inverse: returns false if a == 0 (no inverse).
bool gf16_inv_checked(gf16_t a, gf16_t* out) { if ((a & GF16_MASK) == 0) return false;
	 *out = gf16_inv(a); return true; }
	 
// GF16_SHORT_NAMES 

gf16_t add(gf16_t a, gf16_t b) { return gf16_add(a, b); } 
gf16_t sub(gf16_t a, gf16_t b) { return gf16_sub(a, b); } 
gf16_t mult(gf16_t a, gf16_t b) { return gf16_mult(a, b); } 
gf16_t inv(gf16_t a) { return gf16_inv(a); } 


void gf16_init(void) { 
	if (gf16__initialized) return;
	gf16_t exp_tbl[15];      // a^0 .. a^14
	unsigned char log_tbl[16]; // log base a; 0 is undefined (0xFF)

	// Build exp/log tables using primitive polynomial x^4 + x + 1.
	exp_tbl[0] = 1;
	for (int i = 1; i < 15; ++i)
	    exp_tbl[i] = gf16_xtime(exp_tbl[i - 1]);

	for (int i = 0; i < 16; ++i) log_tbl[i] = 0xFF;
	for (int i = 0; i < 15; ++i) log_tbl[exp_tbl[i]] = (unsigned char)i;

	// Inverse table
	gf16_inv_table[0] = 0; // by convention; 0 has no multiplicative inverse
	for (int a = 1; a < 16; ++a) {
	    unsigned char la = log_tbl[a];
	    // inv(a) = a^(15 - la) = exp_tbl[(15 - la) % 15]
	    gf16_inv_table[a] = exp_tbl[(15 - la) % 15];
	}

	// Multiplication table
	for (int a = 0; a < 16; ++a) {
	    for (int b = 0; b < 16; ++b) {
	        if (a == 0 || b == 0) {
	            gf16_mul_table[a][b] = 0;
	        } else {
	            unsigned char la = log_tbl[a];
	            unsigned char lb = log_tbl[b];
	            gf16_mul_table[a][b] = exp_tbl[(la + lb) % 15];
	        }
	    }
	}

	gf16__initialized = 1;
}
