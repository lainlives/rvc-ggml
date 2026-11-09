// npy.cpp -- M4-5a implementation (see the npy.h header comment)

#include "export/npy.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace rvc_train::exporter {
namespace {

bool read_file_bytes(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        if (err) {
            *err = "cannot open: " + path;
        }
        return false;
    }
    const std::streamoff size = f.tellg();
    f.seekg(0, std::ios::beg);
    out->resize(static_cast<std::size_t>(size));
    if (size > 0 && !f.read(reinterpret_cast<char*>(out->data()), size)) {
        if (err) {
            *err = "read failed: " + path;
        }
        return false;
    }
    return true;
}

// Find the value fragment of `'key': value` in the header text (excluding the key)
bool find_field(const std::string& header, const char* key, std::string* value) {
    const std::string pat = std::string("'") + key + "'";
    const std::size_t k = header.find(pat);
    if (k == std::string::npos) {
        return false;
    }
    const std::size_t colon = header.find(':', k + pat.size());
    if (colon == std::string::npos) {
        return false;
    }
    std::size_t begin = colon + 1;
    while (begin < header.size() && std::isspace(static_cast<unsigned char>(header[begin]))) {
        ++begin;
    }
    std::size_t end = begin;
    if (begin < header.size() && header[begin] == '(') {
        end = header.find(')', begin);
        if (end == std::string::npos) {
            return false;
        }
        ++end;
    } else {
        while (end < header.size() && header[end] != ',' && header[end] != '}') {
            ++end;
        }
    }
    *value = header.substr(begin, end - begin);
    return true;
}

bool parse_shape(const std::string& text, int64_t* rows, int64_t* cols, std::string* err) {
    if (text.empty() || text[0] != '(') {
        if (err) {
            *err = "npy shape parse failed: " + text;
        }
        return false;
    }
    std::vector<int64_t> dims;
    std::stringstream ss(text.substr(1));
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Strip whitespace
        const std::size_t b = item.find_first_not_of(" \t");
        if (b == std::string::npos) {
            continue;
        }
        const std::size_t e = item.find_last_not_of(" \t");
        const std::string tok = item.substr(b, e - b + 1);
        try {
            dims.push_back(std::stoll(tok));
        } catch (const std::exception&) {
            if (err) {
                *err = "invalid npy shape dimension: " + tok;
            }
            return false;
        }
    }
    if (dims.empty() || dims.size() > 2) {
        if (err) {
            *err = "npy supports only 1D/2D arrays";
        }
        return false;
    }
    *rows = dims[0];
    *cols = dims.size() == 2 ? dims[1] : 1;
    return true;
}

} // namespace

bool load_npy_f32(const std::string& path, std::vector<float>* out, int64_t* rows, int64_t* cols,
                  std::string* err) {
    out->clear();
    *rows = 0;
    *cols = 0;
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(path, &bytes, err)) {
        return false;
    }
    if (bytes.size() < 10 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0) {
        if (err) {
            *err = "not a .npy file (bad magic): " + path;
        }
        return false;
    }
    const int major = bytes[6];
    std::size_t pos = 8;
    std::size_t header_len = 0;
    if (major == 1) {
        if (bytes.size() < pos + 2) {
            if (err) {
                *err = "truncated npy header: " + path;
            }
            return false;
        }
        header_len = static_cast<std::size_t>(bytes[pos]) |
                     (static_cast<std::size_t>(bytes[pos + 1]) << 8);
        pos += 2;
    } else if (major == 2 || major == 3) {
        if (bytes.size() < pos + 4) {
            if (err) {
                *err = "truncated npy header: " + path;
            }
            return false;
        }
        header_len = static_cast<std::size_t>(bytes[pos]) |
                     (static_cast<std::size_t>(bytes[pos + 1]) << 8) |
                     (static_cast<std::size_t>(bytes[pos + 2]) << 16) |
                     (static_cast<std::size_t>(bytes[pos + 3]) << 24);
        pos += 4;
    } else {
        if (err) {
            *err = "unsupported npy version: " + std::to_string(major);
        }
        return false;
    }
    if (bytes.size() < pos + header_len) {
        if (err) {
            *err = "npy header length out of bounds: " + path;
        }
        return false;
    }
    const std::string header(reinterpret_cast<const char*>(bytes.data() + pos), header_len);
    pos += header_len;

    std::string descr;
    std::string fortran;
    std::string shape;
    if (!find_field(header, "descr", &descr) || !find_field(header, "fortran_order", &fortran) ||
        !find_field(header, "shape", &shape)) {
        if (err) {
            *err = "npy header missing descr/fortran_order/shape: " + path;
        }
        return false;
    }
    if (descr.size() >= 2 && descr.front() == '\'' && descr.back() == '\'') {
        descr = descr.substr(1, descr.size() - 2);
    }
    if (descr != "<f4" && descr != "=f4" && descr != "|f4") {
        if (err) {
            *err = "npy dtype supports only F32 (<f4): " + descr + " @ " + path;
        }
        return false;
    }
    if (fortran.find("True") != std::string::npos) {
        if (err) {
            *err = "npy fortran_order=True is not supported (C order required): " + path;
        }
        return false;
    }
    if (!parse_shape(shape, rows, cols, err)) {
        return false;
    }
    const int64_t n = *rows * *cols;
    if (n < 0 || static_cast<std::size_t>(n) * 4 > bytes.size() - pos) {
        if (err) {
            *err = "npy data length too short: " + path;
        }
        return false;
    }
    out->resize(static_cast<std::size_t>(n));
    if (n > 0) {
        std::memcpy(out->data(), bytes.data() + pos, static_cast<std::size_t>(n) * 4);
    }
    return true;
}

bool load_npy_dir_f32(const std::string& dir, std::vector<float>* out, int64_t* rows,
                      int64_t* cols, std::string* err) {
    out->clear();
    *rows = 0;
    *cols = 0;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        if (err) {
            *err = "not a directory: " + dir;
        }
        return false;
    }
    std::vector<std::string> names;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (entry.is_regular_file(ec)) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".npy") {
                names.push_back(entry.path().string());
            }
        }
    }
    std::sort(names.begin(), names.end());
    if (names.empty()) {
        if (err) {
            *err = "no .npy files in directory: " + dir;
        }
        return false;
    }
    for (const std::string& p : names) {
        std::vector<float> part;
        int64_t r = 0;
        int64_t c = 0;
        if (!load_npy_f32(p, &part, &r, &c, err)) {
            return false;
        }
        if (*cols == 0) {
            *cols = c;
        } else if (c != *cols) {
            if (err) {
                *err = "inconsistent .npy column count in directory: " + p;
            }
            return false;
        }
        out->insert(out->end(), part.begin(), part.end());
        *rows += r;
    }
    return true;
}

} // namespace rvc_train::exporter
