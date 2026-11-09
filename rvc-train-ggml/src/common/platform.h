// platform.h - small cross-platform helpers shared by the rvc-train modules:
// environment variables and the native path separator (MSVC / MinGW / POSIX).
#pragma once

#include <cstdlib>
#include <string>

namespace rvc_train {
namespace platform {

// True when the environment variable is set (even to an empty value).
inline bool env_present(const char* name) {
#if defined(_MSC_VER)
    // getenv triggers C4996 under MSVC /W4; keep the _dupenv_s path there.
    char*  value = nullptr;
    size_t size  = 0;
    if (_dupenv_s(&value, &size, name) == 0 && value != nullptr) {
        std::free(value);
        return true;
    }
    return false;
#else
    return std::getenv(name) != nullptr;
#endif
}

// Value of the environment variable, or an empty string when it is not set.
inline std::string env_get(const char* name) {
#if defined(_MSC_VER)
    char*  value = nullptr;
    size_t size  = 0;
    if (_dupenv_s(&value, &size, name) == 0 && value != nullptr) {
        std::string out(value);
        std::free(value);
        return out;
    }
    return std::string();
#else
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::string();
#endif
}

// Set an environment variable, overwriting any previous value.
inline void env_set(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

// Native path separator for building human-readable paths in logs and messages.
// (Runtime file access accepts '/' on every supported platform.)
inline char path_sep() {
#if defined(_WIN32)
    return '\\';
#else
    return '/';
#endif
}

} // namespace platform
} // namespace rvc_train
