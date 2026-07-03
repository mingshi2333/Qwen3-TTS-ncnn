// Tokenizer parity: C++ QwenBpe vs HF ids on the mixed corpus produced by
// export/export_tokenizer.py (tests/data/tokenizer_test.bin).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "qwen_bpe.h"

using namespace q3tts;

int main(int argc, char** argv) {
    const std::string tok_file = argc > 1 ? argv[1] : "models/tokenizer.txt";
    const std::string test_file = argc > 2 ? argv[2] : "tests/data/tokenizer_test.bin";

    QwenBpe bpe;
    bpe.load(tok_file);

    std::ifstream f(test_file, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", test_file.c_str()); return 1; }
    uint32_t count;
    f.read((char*)&count, 4);

    int failed = 0;
    for (uint32_t c = 0; c < count; c++) {
        uint32_t tl, nl;
        f.read((char*)&tl, 4);
        std::string text(tl, '\0');
        f.read(text.data(), tl);
        f.read((char*)&nl, 4);
        std::vector<int> ref(nl);
        f.read((char*)ref.data(), nl * 4);

        std::vector<int> got = bpe.encode(text);
        if (got != ref) {
            failed++;
            if (failed <= 5) {
                printf("case %u MISMATCH: \"%.60s\"\n  got:", c, text.c_str());
                for (size_t i = 0; i < got.size() && i < 24; i++) printf(" %d", got[i]);
                printf("\n  ref:");
                for (size_t i = 0; i < ref.size() && i < 24; i++) printf(" %d", ref[i]);
                printf("\n");
            }
        }
    }
    printf("tokenizer parity: %u/%u cases match %s\n", count - failed, count,
           failed == 0 ? "[PASS]" : "[FAIL]");
    return failed == 0 ? 0 : 1;
}
