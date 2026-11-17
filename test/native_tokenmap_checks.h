#include "native_tokenmap_reference.h"
static void test_native_tokenmap(void) {
    BnJSON doc={0};assert(!bn_json_init(&doc,native_tokenmap_fixture_json,strlen(native_tokenmap_fixture_json)));
    BnTokenizerVocab v={0};size_t vb;assert(!bn_tokenizer_vocab_plan(NATIVE_TOKENMAP_VOCAB,&vb)&&!bn_tokenizer_vocab_init_json(&v,&doc,NATIVE_TOKENMAP_VOCAB,vb));
    BnTokenizerMapSpec spec={BN_TOKENIZER_MAP_ENGRAM_V41,NATIVE_TOKENMAP_VOCAB,NATIVE_TOKENMAP_COMPRESSED,NATIVE_TOKENMAP_PAD};
    BnTokenizerMapWorkspace w,saved;memset(&saved,0xa5,sizeof(saved));w=saved;
    BnTokenizerMapSpec bad=spec;bad.profile=BN_TOKENIZER_MAP_NONE;
    assert(bn_tokenizer_map_plan(&v,&bad,&w)==-1&&!memcmp(&w,&saved,sizeof(w)));
    assert(!bn_tokenizer_map_plan(&v,&spec,&w)&&w.host_bytes&&w.hash_slots>=(size_t)spec.vocab_size*2);
    BnTokenizerMap map={0};assert(bn_tokenizer_map_init(&map,&v,&spec,w.host_bytes-1)==-1&&!map.state);
    bad=spec;bad.compressed_vocab--;assert(bn_tokenizer_map_init(&map,&v,&bad,w.host_bytes)==-1&&!map.state);
    bad=spec;bad.pad_token=bad.vocab_size;assert(bn_tokenizer_map_init(&map,&v,&bad,w.host_bytes)==-1&&!map.state);
    assert(!bn_tokenizer_map_init(&map,&v,&spec,w.host_bytes));void *state=map.state;
    assert(bn_tokenizer_map_init(&map,&v,&spec,w.host_bytes)==-1&&map.state==state);
    BnTokenizerMapInfo info;assert(!bn_tokenizer_map_info(&map,&info)&&info.vocab_size==spec.vocab_size&&info.compressed_vocab==spec.compressed_vocab&&info.mapped_pad==native_tokenmap_ids[spec.pad_token]&&info.host_bytes==w.host_bytes);
    assert(!memcmp(bn_tokenizer_map_ids(&map),native_tokenmap_ids,sizeof(native_tokenmap_ids)));
    bn_tokenizer_vocab_free(&v);bn_json_free(&doc);
    /* All retained keys and IDs are owned by the map, including embedded NUL. */
    for(int i=0;i<NATIVE_TOKENMAP_VOCAB;i++) {
        BnTokenizerMapKey key;assert(!bn_tokenizer_map_key(&map,i,&key));
        assert(key.bytes==native_tokenmap_keys[i].bytes&&!memcmp(key.data,native_tokenmap_keys[i].data,key.bytes));
    }
    BnTokenizerMapKey key={(const unsigned char*)"unchanged",99};const unsigned char *old=key.data;
    assert(bn_tokenizer_map_key(&map,-1,&key)==-1&&key.data==old&&key.bytes==99);
    assert(bn_tokenizer_map_key(&map,NATIVE_TOKENMAP_VOCAB,&key)==-1&&key.data==old&&key.bytes==99);
    bn_tokenizer_map_free(&map);bn_tokenizer_map_free(&map);assert(!map.state&&!bn_tokenizer_map_ids(&map));
    puts("Native source Engram token map: Unicode versions, raw/empty fallbacks, counted keys and owned storage PASSED");
}
