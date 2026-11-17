/* Original Rust tokenizers 0.23.3 ByteLevel outputs; regenerate with reference_native_vocab.py. */
static const char native_vocab_fixture_json[] = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"\\u0100\":0,\"\\u00c3\":1,\"\\u00c3\\u00a9\":2,\"\\u0120hi\":3,\"\\u00e2\\u0124\":4,\"\\u00f0\\u0141\\u0134\":5,\"\\u00ef\\u00bf\\u00bd\":6,\"\":7,\"\\ud83d\\ude00\":8,\"<\\uff5c\\u2581pad\\u2581\\uff5c>\":9}},\"added_tokens\":[{\"id\":0,\"content\":\"\\u0100\"},{\"id\":10,\"content\":\"\\u00a0\"},{\"id\":11,\"content\":\"<|text|>\"}],\"decoder\":{\"type\":\"ByteLevel\"}}";
static const unsigned char native_vocab_raw_0[] = {196,128,0};
static const unsigned char native_vocab_decoded_0[] = {0,0};
static const unsigned char native_vocab_raw_1[] = {195,131,0};
static const unsigned char native_vocab_decoded_1[] = {239,191,189,0};
static const unsigned char native_vocab_raw_2[] = {195,131,194,169,0};
static const unsigned char native_vocab_decoded_2[] = {195,169,0};
static const unsigned char native_vocab_raw_3[] = {196,160,104,105,0};
static const unsigned char native_vocab_decoded_3[] = {32,104,105,0};
static const unsigned char native_vocab_raw_4[] = {195,162,196,164,0};
static const unsigned char native_vocab_decoded_4[] = {239,191,189,0};
static const unsigned char native_vocab_raw_5[] = {195,176,197,129,196,180,0};
static const unsigned char native_vocab_decoded_5[] = {239,191,189,0};
static const unsigned char native_vocab_raw_6[] = {195,175,194,191,194,189,0};
static const unsigned char native_vocab_decoded_6[] = {239,191,189,0};
static const unsigned char native_vocab_raw_7[] = {0};
static const unsigned char native_vocab_decoded_7[] = {0};
static const unsigned char native_vocab_raw_8[] = {240,159,152,128,0};
static const unsigned char native_vocab_decoded_8[] = {240,159,152,128,0};
static const unsigned char native_vocab_raw_9[] = {60,239,189,156,226,150,129,112,97,100,226,150,129,239,189,156,62,0};
static const unsigned char native_vocab_decoded_9[] = {60,239,189,156,226,150,129,112,97,100,226,150,129,239,189,156,62,0};
static const unsigned char native_vocab_raw_10[] = {194,160,0};
static const unsigned char native_vocab_decoded_10[] = {194,160,0};
static const unsigned char native_vocab_raw_11[] = {60,124,116,101,120,116,124,62,0};
static const unsigned char native_vocab_decoded_11[] = {60,124,116,101,120,116,124,62,0};
typedef struct {const unsigned char *raw,*decoded;size_t raw_bytes,decoded_bytes;int replacement;} NativeVocabReference;
static const NativeVocabReference native_vocab_reference[] = {
    {native_vocab_raw_0,native_vocab_decoded_0,2,1,0},
    {native_vocab_raw_1,native_vocab_decoded_1,2,3,1},
    {native_vocab_raw_2,native_vocab_decoded_2,4,2,0},
    {native_vocab_raw_3,native_vocab_decoded_3,4,3,0},
    {native_vocab_raw_4,native_vocab_decoded_4,4,3,1},
    {native_vocab_raw_5,native_vocab_decoded_5,6,3,1},
    {native_vocab_raw_6,native_vocab_decoded_6,6,3,1},
    {native_vocab_raw_7,native_vocab_decoded_7,0,0,0},
    {native_vocab_raw_8,native_vocab_decoded_8,4,4,0},
    {native_vocab_raw_9,native_vocab_decoded_9,17,17,0},
    {native_vocab_raw_10,native_vocab_decoded_10,2,2,0},
    {native_vocab_raw_11,native_vocab_decoded_11,8,8,0},
};
