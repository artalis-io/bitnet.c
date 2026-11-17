/* Portable Engram layout RNG, adapted from NumPy 2.3.5 PCG64 and SeedSequence.
 * SeedSequence copyright (c) 2015 Melissa E. O'Neill and (c) 2019 NumPy Developers,
 * under the MIT license reproduced below. No compiler 128-bit extensions. */
/*
 * PCG64 Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 * Copyright 2015 Robert Kern <robert.kern@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     https://www.pcg-random.org
 *
 * Relicensed MIT in May 2019
 *
 * The MIT License
 *
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdint.h>
typedef struct {uint64_t hi,lo;} NativeHashWide;
typedef struct {NativeHashWide state,inc;uint32_t cached;int has_cached;} NativeHashPCG;
static NativeHashWide native_hash_mul64(uint64_t a,uint64_t b) {
    uint64_t a0=(uint32_t)a,a1=a>>32,b0=(uint32_t)b,b1=b>>32;
    uint64_t w=a0*b0,t=a1*b0+(w>>32),x=(uint32_t)t+a0*b1;
    return (NativeHashWide){a1*b1+(t>>32)+(x>>32),a*b};
}
static NativeHashWide native_hash_wide_add(NativeHashWide a,NativeHashWide b) {uint64_t lo=a.lo+b.lo;return (NativeHashWide){a.hi+b.hi+(lo<a.lo),lo};}
static void native_hash_step(NativeHashPCG *g) {
    const uint64_t hi=UINT64_C(2549297995355413924),lo=UINT64_C(4865540595714422341);
    NativeHashWide product=native_hash_mul64(g->state.lo,lo);product.hi+=g->state.hi*lo+g->state.lo*hi;g->state=native_hash_wide_add(product,g->inc);
}
static uint32_t native_hash_seed_mix(uint32_t x,uint32_t *c) {x^=*c;*c*=UINT32_C(0x931e8875);x*=*c;return x^(x>>16);}
static void native_hash_seed(NativeHashPCG *g,uint32_t seed) {
    uint32_t c=UINT32_C(0x43b0d7e5),pool[4],words[8];
    for(int i=0;i<4;i++)pool[i]=native_hash_seed_mix(i?0:seed,&c);
    for(int src=0;src<4;src++)for(int dst=0;dst<4;dst++)if(src!=dst) {
        uint32_t v=UINT32_C(0xca01f9dd)*pool[dst]-UINT32_C(0x4973f715)*native_hash_seed_mix(pool[src],&c);pool[dst]=v^(v>>16);
    }
    c=UINT32_C(0x8b51f9dd);
    for(int i=0;i<8;i++){uint32_t v=pool[i%4]^c;c*=UINT32_C(0x58f38ded);v*=c;words[i]=v^(v>>16);}
    uint64_t w[4];for(int i=0;i<4;i++)w[i]=(uint64_t)words[2*i]|((uint64_t)words[2*i+1]<<32);
    g->has_cached=0;g->cached=0;
    g->state=(NativeHashWide){0,0};g->inc=(NativeHashWide){(w[2]<<1)|(w[3]>>63),(w[3]<<1)|1};native_hash_step(g);
    g->state=native_hash_wide_add(g->state,(NativeHashWide){w[0],w[1]});native_hash_step(g);
}
static uint64_t native_hash_raw(NativeHashPCG *g) {native_hash_step(g);uint64_t v=g->state.hi^g->state.lo;unsigned r=(unsigned)(g->state.hi>>58);return (v>>r)|(v<<((64-r)&63));}
static uint32_t native_hash_raw32(NativeHashPCG *g) {
    if(g->has_cached){g->has_cached=0;return g->cached;}
    uint64_t word=native_hash_raw(g);g->cached=(uint32_t)(word>>32);g->has_cached=1;return (uint32_t)word;
}
static uint64_t native_hash_bounded(NativeHashPCG *g,uint64_t bound) {
    if(bound<=UINT64_C(4294967296)) {
        if(bound==UINT64_C(4294967296))return native_hash_raw32(g);
        uint32_t b=(uint32_t)bound,threshold=(uint32_t)(UINT32_MAX-b+1)%b;uint64_t v;
        do{v=(uint64_t)native_hash_raw32(g)*b;}while((uint32_t)v<threshold);return v>>32;
    }
    uint64_t threshold=(UINT64_MAX-bound+1)%bound;NativeHashWide p;
    do{p=native_hash_mul64(native_hash_raw(g),bound);}while(p.lo<threshold);return p.hi;
}
static int native_hash_prime(uint32_t value) {
    if(value<2)return 0;
    if(!(value&1))return value==2;
    for(uint32_t d=3;d<=value/d;d+=2)if(value%d==0)return 0;
    return 1;
}
