#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "lu_solve.h"

static uint8_t gf16_mul(uint8_t a, uint8_t b) {
    static uint8_t tbl[65536], ok=0;
    if(!ok){
        uint8_t l[16]={0},e[16];e[0]=1;
        for(int i=1;i<15;i++){e[i]=(e[i-1]<<1)^((e[i-1]&8)?0x13:0);l[e[i]&0xF]=i;}
        for(int i=0;i<256;i++)for(int j=0;j<256;j++){
            int a_=i&0xF,b_=i>>4,c_=j&0xF,d_=j>>4,p=0;
            if(a_&&c_){int s=l[a_]+l[c_];if(s>=15)s-=15;p^=e[s];}
            if(b_&&c_){int s=l[b_]+l[c_];if(s>=15)s-=15;p^=e[s]<<4;}
            if(a_&&d_){int s=l[a_]+l[d_];if(s>=15)s-=15;p^=e[s];}
            if(b_&&d_){int s=l[b_]+l[d_];if(s>=15)s-=15;p^=e[s]<<4;}
            tbl[i*256+j]=p;
        }
        ok=1;
    }
    return tbl[((uint)a<<8)|b];
}

static uint8_t gf256_mul(uint8_t a, uint8_t b) {
    static uint8_t tbl[65536],ok=0;
    if(!ok){
        uint8_t l[256]={0},e[512]={0};e[0]=1;
        for(int i=1;i<255;i++){e[i]=(e[i-1]<<1)^((e[i-1]&0x80)?0x11D:0);l[e[i]]=i;}
        for(int i=0;i<256;i++)for(int j=0;j<256;j++){
            if(i==0||j==0){tbl[i*256+j]=0;continue;}
            int s=l[i]+l[j];if(s>=255)s-=255;
            tbl[i*256+j]=e[s];
        }
        ok=1;
    }
    return tbl[((uint)a<<8)|b];
}

static uint8_t gf4_mul(uint8_t a, uint8_t b) {
    static uint8_t tbl[65536],ok=0;
    if(!ok){
        uint8_t l[4]={0},e[4];e[0]=1;e[1]=2;e[2]=3;e[3]=1;
        l[1]=0;l[2]=1;l[3]=2;
        for(int i=0;i<256;i++)for(int j=0;j<256;j++){
            int a_=i&3,b_=(i>>2)&3,c_=(i>>4)&3,d_=(i>>6)&3;
            int e_=j&3,f_=(j>>2)&3,g_=(j>>4)&3,h_=(j>>6)&3,p=0;
            if(a_&&e_){int s=l[a_]+l[e_];if(s>=3)s-=3;p^=e[s];}
            if(b_&&f_)p^=(e[(l[b_]+l[f_])%3]<<2);
            if(c_&&g_)p^=(e[(l[c_]+l[g_])%3]<<4);
            if(d_&&h_)p^=(e[(l[d_]+l[h_])%3]<<6);
            tbl[i*256+j]=p;
        }
        ok=1;
    }
    return tbl[((uint)a<<8)|b];
}

int test(const char *name, int (*solver)(uint8_t*,uint8_t*,uint8_t*,int,int),
         int max_val, int rows, int cols) {
    srand(54321);
    uint8_t *M=(uint8_t*)malloc((size_t)rows*cols);
    uint8_t *r=(uint8_t*)malloc(rows);
    uint8_t *x=(uint8_t*)calloc(cols,1);
    uint8_t *tx=(uint8_t*)malloc(cols);

    for(int c=0;c<cols;c++) tx[c]=1+rand()%max_val;
    for(int rr=0;rr<rows;rr++){
        uint8_t s=0;
        for(int c=0;c<cols;c++){
            uint8_t v=1+rand()%max_val;
            M[rr*cols+c]=v;
            if(max_val==1) s ^= v & tx[c];
            else if(max_val==3) s ^= gf4_mul(v,tx[c]);
            else if(max_val==15) s ^= gf16_mul(v,tx[c]);
            else s ^= gf256_mul(v,tx[c]);
        }
        r[rr]=s;
    }

    int rk=solver(M,r,x,rows,cols);
    int ok=1;
    for(int c=0;c<cols;c++) if(x[c]!=tx[c]){ok=0;printf("  diff at %d: %x vs %x\n",c,x[c],tx[c]);break;}
    printf("  %-12s %dx%d rank=%-4d %s\n",name,rows,cols,rk,ok?"YES":"NO");
    free(M);free(r);free(x);free(tx);
    return ok;
}

int main() {
    printf("Solver correctness (known-consistent systems):\n");
    test("GF(16)",  solve_lu_gf16,  15, 100,50);
    test("GF(256)", solve_lu_gf256,255, 100,50);
    test("GF(4)",   solve_lu_gf4,   3,  100,50);
    test("Binary",  solve_lu_binary, 1, 100,50);
    return 0;
}
