/* Independent unchanged pinned FLA operators: packed and rsqrt profiles. */
#include <stdint.h>
typedef struct {int batch,seed,mode;uint64_t out[2],state[2];} BnBF16GDNDecodeProfilesReference;
static const BnBF16GDNDecodeProfilesReference bf16_gdn_decode_profiles_cases[]={
{1,7,0,{UINT64_C(0x15ff9e3ac62b5de1),UINT64_C(0x15ff9e3ac62b5de1)},{UINT64_C(0x2da3049c272505bf),UINT64_C(0x8e89b2699b7d76a4)}},
{2,3,0,{UINT64_C(0xdf94fa90ab441cc5),UINT64_C(0xdf94fa90ab441cc5)},{UINT64_C(0x0f475da5c2687715),UINT64_C(0x00a0236f964780b4)}},
{2,11,0,{UINT64_C(0x58dbf92e71a23dfb),UINT64_C(0x58dbf92e71a23dfb)},{UINT64_C(0x5dfc376d770c8750),UINT64_C(0x1aa240b0b42a747d)}},
{4,13,0,{UINT64_C(0xbd6b05dc6df7cd7d),UINT64_C(0xbd6b05dc6df7cd7d)},{UINT64_C(0xf987ec147ba6d3e5),UINT64_C(0xbfbb345fcd685a95)}},
{1,17,1,{UINT64_C(0x9590bfd690920ae9),UINT64_C(0x9590bfd690920ae9)},{UINT64_C(0x8293486e31affe48),UINT64_C(0xe20a6e2c8753ced8)}},
{2,19,1,{UINT64_C(0x0e6c35ca47c3ee73),UINT64_C(0x74b72cbde1d9a8ff)},{UINT64_C(0x0efd0589e0318dfd),UINT64_C(0x65e8ac153ca2316a)}},
{3,23,1,{UINT64_C(0x6184aa4949bc0298),UINT64_C(0x6184aa4949bc0298)},{UINT64_C(0xe8e5c8ce36130cbf),UINT64_C(0x80aac4b66e4e73d6)}},
{4,29,1,{UINT64_C(0x685080613434a2bd),UINT64_C(0x685080613434a2bd)},{UINT64_C(0xf479a3999ae2aaab),UINT64_C(0x0fae439ebb703547)}},
};
