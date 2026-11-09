// utils.h
// General-purpose utilities: logging, timing, string handling, filesystem helpers, etc.
// Design modeled on the same-named module in MSST-GGML (independent implementation here, trimmed down for RVC's needs).

#pragma once

// Remove macro pollution introduced by windows.h
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
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <filesystem>

// --------------------------------------------------------------
// Lightweight high-resolution timer
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
// Simple console logger
// --------------------------------------------------------------
class Logger {
public:
    enum Level { INFO, WARN, ERROR };

    static void set_level(Level level) { min_level_ = level; }

    static void info(const std::string& msg) { log(INFO, msg); }
    static void warn(const std::string& msg) { log(WARN, msg); }
    static void error(const std::string& msg) { log(ERROR, msg); }

    // Simple progress bar: [====>     ] 50.0%
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
// String / filesystem utilities
// --------------------------------------------------------------
namespace utils {
    std::string trim(const std::string& str);
    std::vector<std::string> split(const std::string& str, char delimiter);
    bool file_exists(const std::string& path);
    std::string stem(const std::string& filepath);
    std::vector<std::string> list_files(const std::string& directory, const std::string& extension);
    std::string filename(const std::string& filepath);
    bool ensure_directory(const std::string& dir);

    // Set an environment variable, overwriting any previous value (portable wrapper: _putenv_s / setenv).
    inline void set_env(const char* name, const char* value) {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }
}
