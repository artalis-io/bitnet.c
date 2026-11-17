#include "platform.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    const size_t size = 8u * 1024u * 1024u + 97u;
    unsigned char *data = bn_platform_aligned_alloc(32, size);
    assert(data);
    for (size_t i = 0; i < size; i++)
        data[i] = (unsigned char)(i * 37u + 11u);

    /* Unaligned subspans must preserve their contents and surrounding bytes.
     * Hinting after first touch also catches accidental discard advice. */
    bn_platform_advise_huge_pages(data + 1, size - 2);
    bn_platform_advise_huge_pages(data + 33, size - 66);
    bn_platform_advise_huge_pages(NULL, size);
    bn_platform_advise_huge_pages(data, 0);
    bn_platform_advise_huge_pages(data, 31);
    bn_platform_advise_huge_pages(data, SIZE_MAX);
    for (size_t i = 0; i < size; i++)
        assert(data[i] == (unsigned char)(i * 37u + 11u));
    bn_platform_aligned_free(data);
    puts("Platform huge-page hint: unaligned spans, content/guard preservation, small/null/overflow no-ops PASSED");
    return 0;
}
