/* Independent Rust full-decode prefixes, including early stop/flush. */
static const int native_text_control_ids[]={195,313,169,0,237,160};
static const struct {int skip,limit;const unsigned char *bytes;size_t count;} native_text_control_cases[]={
{0,0,(const unsigned char*)"",0},
{0,1,(const unsigned char*)"\xef\xbf\xbd",3},
{0,2,(const unsigned char*)"\xef\xbf\xbd\x3c\x74\x61\x67\x3e",8},
{0,3,(const unsigned char*)"\xef\xbf\xbd\x3c\x74\x61\x67\x3e\xef\xbf\xbd",11},
{0,4,(const unsigned char*)"\xef\xbf\xbd\x3c\x74\x61\x67\x3e\xef\xbf\xbd\x00",12},
{0,5,(const unsigned char*)"\xef\xbf\xbd\x3c\x74\x61\x67\x3e\xef\xbf\xbd\x00\xef\xbf\xbd",15},
{0,6,(const unsigned char*)"\xef\xbf\xbd\x3c\x74\x61\x67\x3e\xef\xbf\xbd\x00\xef\xbf\xbd\xef\xbf\xbd",18},
{1,0,(const unsigned char*)"",0},
{1,1,(const unsigned char*)"\xef\xbf\xbd",3},
{1,2,(const unsigned char*)"\xef\xbf\xbd",3},
{1,3,(const unsigned char*)"\xc3\xa9",2},
{1,4,(const unsigned char*)"\xc3\xa9\x00",3},
{1,5,(const unsigned char*)"\xc3\xa9\x00\xef\xbf\xbd",6},
{1,6,(const unsigned char*)"\xc3\xa9\x00\xef\xbf\xbd\xef\xbf\xbd",9},
};
