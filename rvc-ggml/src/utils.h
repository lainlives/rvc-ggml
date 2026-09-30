// utils.h
// 通用工具函数集：日志、计时、字符串处理、文件系统辅助等。
// 设计参照 MSST-GGML 同名模块（本项目独立实现，按 RVC 需要精简）。

#pragma once

// 清除 windows.h 带来的宏污染
#ifdef ERROR
#undef ERROR
#endif
#ifdef INFO
#undef INFO
#endif

#include <string>
#include <vector>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <filesystem>

// --------------------------------------------------------------
// 轻量级高精度计时器
// --------------------------------------------------------------
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() { start_ = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

    std::string elapsed_str() const {
        double sec = elapsed_ms() / 1000.0;
        int total_sec = static_cast<int>(sec);
        int minutes = total_sec / 60;
        int seconds = total_sec % 60;
        std::stringstream ss;
        if (minutes > 0) ss << minutes << "m ";
        ss << seconds << "s";
        return ss.str();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// --------------------------------------------------------------
// 简单的控制台日志器
// --------------------------------------------------------------
class Logger {
public:
    enum Level { INFO, WARN, ERROR };

    static void set_level(Level level) { min_level_ = level; }

    static void info(const std::string& msg) { log(INFO, msg); }
    static void warn(const std::string& msg) { log(WARN, msg); }
    static void error(const std::string& msg) { log(ERROR, msg); }

    // 简易进度条：[====>     ] 50.0%
    static void progress(int current, int total, const std::string& desc = "") {
        if (total <= 0) return;
        const int bar_width = 20;
        float ratio = static_cast<float>(current) / total;
        int filled = static_cast<int>(bar_width * ratio);
        std::string bar(bar_width, ' ');
        for (int i = 0; i < filled; ++i) bar[i] = '=';
        if (filled < bar_width) bar[filled] = '>';
        std::stringstream ss;
        ss << "\r[" << bar << "] " << std::fixed << std::setprecision(1) << ratio * 100.0 << "%";
        if (!desc.empty()) ss << " - " << desc;
        std::cout << ss.str() << std::flush;
        if (current == total) std::cout << std::endl;
    }

private:
    static void log(Level level, const std::string& msg) {
        if (level < min_level_) return;
        std::string prefix;
        switch (level) {
        case INFO:  prefix = "[INFO] "; break;
        case WARN:  prefix = "[WARN] "; break;
        case ERROR: prefix = "[ERROR] "; break;
        }
        std::cerr << prefix << msg << std::endl;
    }

    static Level min_level_;
};

// --------------------------------------------------------------
// 字符串 / 文件系统工具
// --------------------------------------------------------------
namespace utils {
    std::string trim(const std::string& str);
    std::vector<std::string> split(const std::string& str, char delimiter);
    bool file_exists(const std::string& path);
    std::string stem(const std::string& filepath);
    std::vector<std::string> list_files(const std::string& directory, const std::string& extension);
    std::string filename(const std::string& filepath);
    bool ensure_directory(const std::string& dir);
}
