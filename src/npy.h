// Minimal .npy reader — just enough for our test vectors and model aux data:
// little-endian '<f4' / '<i4', C-contiguous, version 1.0/2.0 headers.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace q3tts {

struct NpyArray {
    std::vector<int> shape;
    std::vector<char> data;  // raw bytes
    bool is_int = false;     // '<i4' vs '<f4'

    const float* f32() const { return reinterpret_cast<const float*>(data.data()); }
    const int32_t* i32() const { return reinterpret_cast<const int32_t*>(data.data()); }
    size_t numel() const {
        size_t n = 1;
        for (int d : shape) n *= (size_t)d;
        return n;
    }
};

inline NpyArray load_npy(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("npy: cannot open " + path);

    char magic[6];
    if (fread(magic, 1, 6, fp) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npy: bad magic in " + path);
    unsigned char ver[2];
    fread(ver, 1, 2, fp);
    uint32_t header_len = 0;
    if (ver[0] == 1) {
        uint16_t l;
        fread(&l, 2, 1, fp);
        header_len = l;
    } else {
        fread(&header_len, 4, 1, fp);
    }
    std::string header(header_len, '\0');
    fread(header.data(), 1, header_len, fp);

    NpyArray arr;
    if (header.find("'<f4'") != std::string::npos) arr.is_int = false;
    else if (header.find("'<i4'") != std::string::npos) arr.is_int = true;
    else throw std::runtime_error("npy: unsupported dtype in " + path + ": " + header);
    if (header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("npy: need C-order in " + path);

    size_t p = header.find("'shape': (");
    size_t q = header.find(')', p);
    std::string dims = header.substr(p + 10, q - p - 10);
    for (size_t i = 0; i < dims.size();) {
        while (i < dims.size() && !isdigit(dims[i])) i++;
        if (i >= dims.size()) break;
        arr.shape.push_back(atoi(dims.c_str() + i));
        while (i < dims.size() && isdigit(dims[i])) i++;
    }
    if (arr.shape.empty()) arr.shape.push_back(1);

    arr.data.resize(arr.numel() * 4);
    if (fread(arr.data.data(), 1, arr.data.size(), fp) != arr.data.size())
        throw std::runtime_error("npy: short read in " + path);
    fclose(fp);
    return arr;
}

}  // namespace q3tts
