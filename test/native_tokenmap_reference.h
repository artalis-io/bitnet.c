/* Unchanged source Engram map plus Rust0.23.3 key goldens. */
static const char native_tokenmap_fixture_json[] = "{\n  \"version\": \"1.0\",\n  \"truncation\": null,\n  \"padding\": null,\n  \"added_tokens\": [],\n  \"normalizer\": null,\n  \"pre_tokenizer\": null,\n  \"post_processor\": null,\n  \"decoder\": {\n    \"type\": \"ByteLevel\",\n    \"add_prefix_space\": true,\n    \"trim_offsets\": true,\n    \"use_regex\": true\n  },\n  \"model\": {\n    \"type\": \"BPE\",\n    \"dropout\": null,\n    \"unk_token\": null,\n    \"continuing_subword_prefix\": null,\n    \"end_of_word_suffix\": null,\n    \"fuse_unk\": false,\n    \"byte_fallback\": false,\n    \"ignore_merges\": false,\n    \"vocab\": {\n      \"A\": 0,\n      \"a\": 1,\n      \"\u00c3\u012b\": 2,\n      \"e\": 3,\n      \"\u00c3\u00a9\": 4,\n      \"\u0120\": 5,\n      \"\u0120\u0120\": 6,\n      \"\u0109\u010a\": 7,\n      \"\u00cc\u0123\": 8,\n      \"\u00e0\u00a6\u00be\": 9,\n      \"\u0120x\u0120\": 10,\n      \"x\": 11,\n      \"\u00ee\u0122\u0122a\": 12,\n      \"\u0120a\": 13,\n      \"\u00ce\u0141\u00ce\u00a3\": 14,\n      \"\u00ce\u00bf\u00cf\u0125\": 15,\n      \"\u00ce\u00bf\u00cf\u0124\": 16,\n      \"\u0100\": 17,\n      \"\": 18,\n      \"\u0100a\": 19,\n      \"\u0100b\": 20,\n      \"\u0120\u00c2\u0127\": 21,\n      \"\u00c2\u0127\": 22,\n      \"\u00ee\u0122\u0122\": 23,\n      \"a\u00ee\u0122\u0122\u0120\": 24,\n      \"a\u0120\": 25,\n      \"\u00e1\u00b2\u012b\": 26,\n      \"\u00e1\u00b2\u012c\": 27,\n      \"A\u00e1\u00aa\u00bf\": 28,\n      \"A\u00cc\u0123\": 29,\n      \"\u00f0\u013e\u00b3\u0138\": 30,\n      \"\u00c3\": 31,\n      \"\u00ef\u00bf\u00bd\": 32\n    },\n    \"merges\": []\n  }\n}";
enum {NATIVE_TOKENMAP_VOCAB=33,NATIVE_TOKENMAP_COMPRESSED=21,NATIVE_TOKENMAP_PAD=5};
static const uint32_t native_tokenmap_ids[] = {0,0,1,1,1,2,2,2,3,4,5,5,6,0,7,7,8,9,10,11,12,13,14,2,15,0,16,16,17,0,18,19,20};
static const unsigned char native_tokenmap_key_0[] = {97,0};
static const unsigned char native_tokenmap_key_1[] = {97,0};
static const unsigned char native_tokenmap_key_2[] = {101,0};
static const unsigned char native_tokenmap_key_3[] = {101,0};
static const unsigned char native_tokenmap_key_4[] = {101,0};
static const unsigned char native_tokenmap_key_5[] = {32,0};
static const unsigned char native_tokenmap_key_6[] = {32,0};
static const unsigned char native_tokenmap_key_7[] = {32,0};
static const unsigned char native_tokenmap_key_8[] = {204,129,0};
static const unsigned char native_tokenmap_key_9[] = {224,166,190,0};
static const unsigned char native_tokenmap_key_10[] = {120,0};
static const unsigned char native_tokenmap_key_11[] = {120,0};
static const unsigned char native_tokenmap_key_12[] = {32,97,0};
static const unsigned char native_tokenmap_key_13[] = {97,0};
static const unsigned char native_tokenmap_key_14[] = {206,191,207,131,0};
static const unsigned char native_tokenmap_key_15[] = {206,191,207,131,0};
static const unsigned char native_tokenmap_key_16[] = {206,191,207,130,0};
static const unsigned char native_tokenmap_key_17[] = {0,0};
static const unsigned char native_tokenmap_key_18[] = {0};
static const unsigned char native_tokenmap_key_19[] = {0,97,0};
static const unsigned char native_tokenmap_key_20[] = {0,98,0};
static const unsigned char native_tokenmap_key_21[] = {32,194,133,0};
static const unsigned char native_tokenmap_key_22[] = {194,133,0};
static const unsigned char native_tokenmap_key_23[] = {32,0};
static const unsigned char native_tokenmap_key_24[] = {97,32,0};
static const unsigned char native_tokenmap_key_25[] = {97,0};
static const unsigned char native_tokenmap_key_26[] = {225,178,138,0};
static const unsigned char native_tokenmap_key_27[] = {225,178,138,0};
static const unsigned char native_tokenmap_key_28[] = {97,225,170,191,0};
static const unsigned char native_tokenmap_key_29[] = {97,0};
static const unsigned char native_tokenmap_key_30[] = {240,156,179,150,0};
static const unsigned char native_tokenmap_key_31[] = {195,131,0};
static const unsigned char native_tokenmap_key_32[] = {195,175,194,191,194,189,0};
static const struct {const unsigned char *data;size_t bytes;} native_tokenmap_keys[] = {
    {native_tokenmap_key_0,1},
    {native_tokenmap_key_1,1},
    {native_tokenmap_key_2,1},
    {native_tokenmap_key_3,1},
    {native_tokenmap_key_4,1},
    {native_tokenmap_key_5,1},
    {native_tokenmap_key_6,1},
    {native_tokenmap_key_7,1},
    {native_tokenmap_key_8,2},
    {native_tokenmap_key_9,3},
    {native_tokenmap_key_10,1},
    {native_tokenmap_key_11,1},
    {native_tokenmap_key_12,2},
    {native_tokenmap_key_13,1},
    {native_tokenmap_key_14,4},
    {native_tokenmap_key_15,4},
    {native_tokenmap_key_16,4},
    {native_tokenmap_key_17,1},
    {native_tokenmap_key_18,0},
    {native_tokenmap_key_19,2},
    {native_tokenmap_key_20,2},
    {native_tokenmap_key_21,3},
    {native_tokenmap_key_22,2},
    {native_tokenmap_key_23,1},
    {native_tokenmap_key_24,2},
    {native_tokenmap_key_25,1},
    {native_tokenmap_key_26,3},
    {native_tokenmap_key_27,3},
    {native_tokenmap_key_28,4},
    {native_tokenmap_key_29,1},
    {native_tokenmap_key_30,4},
    {native_tokenmap_key_31,2},
    {native_tokenmap_key_32,6},
};
