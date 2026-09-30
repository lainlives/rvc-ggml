// utils.cpp
// 工具函数实现

#include "utils.h"
#include <algorithm>
#include <cctype>

Logger::Level Logger::min_level_ = Logger::INFO;

namespace utils {

    std::string trim(const std::string& str) {
        auto begin = str.begin();
        while (begin != str.end() && std::isspace(static_cast<unsigned char>(*begin))) {
            ++begin;
        }
        auto end = str.end();
        while (end != begin && std::isspace(static_cast<unsigned char>(*(end - 1)))) {
            --end;
        }
        return std::string(begin, end);
    }

    std::vector<std::string> split(const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::stringstream ss(str);
        std::string token;
        while (std::getline(ss, token, delimiter)) {
            tokens.push_back(trim(token));
        }
        return tokens;
    }

    bool file_exists(const std::string& path) {
        std::filesystem::path p(path);
        return std::filesystem::exists(p) && std::filesystem::is_regular_file(p);
    }

    std::string stem(const std::string& filepath) {
        std::filesystem::path p(filepath);
        return p.stem().string();
    }

    std::string filename(const std::string& filepath) {
        std::filesystem::path p(filepath);
        return p.filename().string();
    }

    std::vector<std::string> list_files(const std::string& directory, const std::string& extension) {
        std::vector<std::string> result;
        std::filesystem::path dir(directory);
        if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
            return result;
        }
        std::string lower_ext = extension;
        std::transform(lower_ext.begin(), lower_ext.end(), lower_ext.begin(),
            [](unsigned char c) { return std::tolower(c); });
        if (!lower_ext.empty() && lower_ext[0] != '.') {
            lower_ext = "." + lower_ext;
        }
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            std::string fname = entry.path().filename().string();
            if (extension.empty()) {
                result.push_back(fname);
                continue;
            }
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                [](unsigned char c) { return std::tolower(c); });
            if (ext == lower_ext) {
                result.push_back(fname);
            }
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    bool ensure_directory(const std::string& dir) {
        std::filesystem::path p(dir);
        std::error_code ec;
        if (!std::filesystem::exists(p)) {
            std::filesystem::create_directories(p, ec);
            return !ec;
        }
        return true;
    }

} // namespace utils
