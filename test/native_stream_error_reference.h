/* Independent surrogate-prefix timing/error references; full Rust checked. */
static const struct {int ids[8];size_t count;int skip;const unsigned char *chunks[9];size_t lengths[9];unsigned char pending[9][4];} native_stream_error_cases[]={
{{237,160},2,0,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd"},{0,0,6},{{1,237,0,0},{2,237,160,0},{0,0,0,0}}},
{{237,160},2,1,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd"},{0,0,6},{{1,237,0,0},{2,237,160,0},{0,0,0,0}}},
{{237,191},2,0,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd"},{0,0,6},{{1,237,0,0},{2,237,191,0},{0,0,0,0}}},
{{237,191},2,1,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd"},{0,0,6},{{1,237,0,0},{2,237,191,0},{0,0,0,0}}},
{{237,160,128},3,0,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd",(const unsigned char*)""},{0,0,9,0},{{1,237,0,0},{2,237,160,0},{0,0,0,0},{0,0,0,0}}},
{{237,160,128},3,1,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd",(const unsigned char*)""},{0,0,9,0},{{1,237,0,0},{2,237,160,0},{0,0,0,0},{0,0,0,0}}},
{{237,160,0},3,0,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd\x00",(const unsigned char*)""},{0,0,7,0},{{1,237,0,0},{2,237,160,0},{0,0,0,0},{0,0,0,0}}},
{{237,160,0},3,1,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd\x00",(const unsigned char*)""},{0,0,7,0},{{1,237,0,0},{2,237,160,0},{0,0,0,0},{0,0,0,0}}},
{{237,160,195,169},4,0,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd",(const unsigned char*)"\xc3\xa9",(const unsigned char*)""},{0,0,6,2,0},{{1,237,0,0},{2,237,160,0},{1,195,0,0},{0,0,0,0},{0,0,0,0}}},
{{237,160,195,169},4,1,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd",(const unsigned char*)"\xc3\xa9",(const unsigned char*)""},{0,0,6,2,0},{{1,237,0,0},{2,237,160,0},{1,195,0,0},{0,0,0,0},{0,0,0,0}}},
{{237,160,256,128},4,0,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd",(const unsigned char*)""},{0,0,0,9,0},{{1,237,0,0},{2,237,160,0},{2,237,160,0},{0,0,0,0},{0,0,0,0}}},
{{237,160,256,128},4,1,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd",(const unsigned char*)""},{0,0,0,9,0},{{1,237,0,0},{2,237,160,0},{2,237,160,0},{0,0,0,0},{0,0,0,0}}},
{{237,160,257,128},4,0,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd\x3c\x73\x3e",(const unsigned char*)"\xef\xbf\xbd",(const unsigned char*)""},{0,0,9,3,0},{{1,237,0,0},{2,237,160,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}}},
{{237,160,257,128},4,1,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd",(const unsigned char*)""},{0,0,0,9,0},{{1,237,0,0},{2,237,160,0},{2,237,160,0},{0,0,0,0},{0,0,0,0}}},
{{237,160,259},3,0,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd",(const unsigned char*)""},{0,0,9,0},{{1,237,0,0},{2,237,160,0},{0,0,0,0},{0,0,0,0}}},
{{237,160,259},3,1,{(const unsigned char*)"",(const unsigned char*)"",(const unsigned char*)"\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd",(const unsigned char*)""},{0,0,9,0},{{1,237,0,0},{2,237,160,0},{0,0,0,0},{0,0,0,0}}},
};
