/* Independent pinned vLLM prep -> chunk GDN; see reference_cuda_bf16_gdn_composed.py. */
#ifndef BN_CUDA_BF16_GDN_COMPOSED_REFERENCE_H
#define BN_CUDA_BF16_GDN_COMPOSED_REFERENCE_H
typedef struct { int tokens,seed,n_phases,splits[2]; uint64_t out[2],state[2]; } BnBF16GDNComposedReference;
static const BnBF16GDNComposedReference cuda_bf16_gdn_composed_cases[] = {
 {5,0,1,{5},{UINT64_C(0x5b83a5f5b67b0e39)},{UINT64_C(0x23bf9acda2c80577)}},
 {5,1,1,{5},{UINT64_C(0xb87ff08f005d123d)},{UINT64_C(0xfb8c9041ceb58a1b)}},
 {65,0,2,{33,32},{UINT64_C(0xb6521d46452d9e6f),UINT64_C(0x39721863b4109a3d)},{UINT64_C(0x70387a01fb8a5ff4),UINT64_C(0x5f31a4e8afc694c5)}},
 {65,1,2,{33,32},{UINT64_C(0xf6d23bc7337dd8b1),UINT64_C(0xd02b85cb474865cf)},{UINT64_C(0x9c418cd3c937b137),UINT64_C(0x5a84843ea7de08bb)}},
 {129,0,2,{64,65},{UINT64_C(0xe2301e7f5340e949),UINT64_C(0x9914fca52bd86e5c)},{UINT64_C(0xffe1d2a2c30ee3f3),UINT64_C(0x4fd210e8aaf723fa)}},
 {129,1,2,{64,65},{UINT64_C(0x3cc88c72fb617640),UINT64_C(0xdc150ca1a463b2e7)},{UINT64_C(0x9edc1dd6f06e1711),UINT64_C(0xd892577360b49c71)}},
 {257,0,2,{129,128},{UINT64_C(0xb29ab43d27f159b0),UINT64_C(0x10a41f775eee65a3)},{UINT64_C(0x4fd210e8aaf723fa),UINT64_C(0xacb115a4cb17e946)}},
 {257,1,2,{129,128},{UINT64_C(0x2fc9801eafbb6c5a),UINT64_C(0xe7a2b46d4ba495e2)},{UINT64_C(0xd892577360b49c71),UINT64_C(0x3446fd73c000467c)}},
 {1047,0,2,{512,535},{UINT64_C(0xb3accc343aff24c4),UINT64_C(0x82ab3a8daeb24716)},{UINT64_C(0xbf3dfc723fa42443),UINT64_C(0x156b2af457998720)}},
 {1047,1,2,{512,535},{UINT64_C(0x43de58547dca45eb),UINT64_C(0x94c3d33ee0c1f5bf)},{UINT64_C(0x9e8e913faed87de3),UINT64_C(0xcb4f68c186aff9a5)}},
};
#endif
