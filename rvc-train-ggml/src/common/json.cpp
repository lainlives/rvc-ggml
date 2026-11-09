// RVC-Train-GGML - minimal JSON parser implementation (src/common/json.cpp)
#include "common/json.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>

namespace rvc_train {
namespace common {

namespace {

const JsonValue& null_value() {
    static const JsonValue k_null;
    return k_null;
}

const std::string& empty_string() {
    static const std::string k_empty;
    return k_empty;
}

bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

void append_utf8(std::string* out, uint32_t cp) {
    if (cp <= 0x7Fu) {
        out->push_back((char) cp);
    } else if (cp <= 0x7FFu) {
        out->push_back((char) (0xC0u | (cp >> 6)));
        out->push_back((char) (0x80u | (cp & 0x3Fu)));
    } else {
        out->push_back((char) (0xE0u | (cp >> 12)));
        out->push_back((char) (0x80u | ((cp >> 6) & 0x3Fu)));
        out->push_back((char) (0x80u | (cp & 0x3Fu)));
    }
}

} // namespace

struct JsonParser {
    const char* p = nullptr;
    const char* end = nullptr;
    std::string err;

    bool fail(const std::string& msg) {
        if (err.empty()) {
            err = msg;
        }
        return false;
    }

    void skip_ws() {
        while (p < end) {
            const char c = *p;
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++p;
            } else {
                break;
            }
        }
    }

    bool match(const char* lit) {
        const size_t n = std::strlen(lit);
        if ((size_t) (end - p) < n || std::memcmp(p, lit, n) != 0) {
            return false;
        }
        p += n;
        return true;
    }

    bool parse_value(JsonValue* out) {
        skip_ws();
        if (p >= end) {
            return fail("incomplete value");
        }
        switch (*p) {
            case '{':
                return parse_object(out);
            case '[':
                return parse_array(out);
            case '"':
                out->type_ = JsonValue::Type::String;
                return parse_string(&out->string_);
            case 't':
                if (match("true")) {
                    out->type_ = JsonValue::Type::Bool;
                    out->boolean_ = true;
                    return true;
                }
                return fail("invalid literal (expected true)");
            case 'f':
                if (match("false")) {
                    out->type_ = JsonValue::Type::Bool;
                    out->boolean_ = false;
                    return true;
                }
                return fail("invalid literal (expected false)");
            case 'n':
                if (match("null")) {
                    out->type_ = JsonValue::Type::Null;
                    return true;
                }
                return fail("invalid literal (expected null)");
            default:
                return parse_number(out);
        }
    }

    bool parse_number(JsonValue* out) {
        const char* start = p;
        while (p < end) {
            const char c = *p;
            if (is_digit(c) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
                ++p;
            } else {
                break;
            }
        }
        if (p == start) {
            return fail(std::string("invalid character: ") + (p < end ? *p : '\0'));
        }
        const std::string text(start, (size_t) (p - start));
        char* endp = nullptr;
        const double v = std::strtod(text.c_str(), &endp);
        if (endp == nullptr || *endp != '\0') {
            return fail("invalid number: " + text);
        }
        out->type_ = JsonValue::Type::Number;
        out->number_ = v;
        return true;
    }

    bool parse_string(std::string* out) {
        if (p >= end || *p != '"') {
            return fail("string is missing the opening quote");
        }
        ++p;
        out->clear();
        while (p < end) {
            const char c = *p++;
            if (c == '"') {
                return true;
            }
            if (c != '\\') {
                out->push_back(c);
                continue;
            }
            if (p >= end) {
                return fail("incomplete string escape");
            }
            const char e = *p++;
            switch (e) {
                case '"':  out->push_back('"');  break;
                case '\\': out->push_back('\\'); break;
                case '/':  out->push_back('/');  break;
                case 'b':  out->push_back('\b'); break;
                case 'f':  out->push_back('\f'); break;
                case 'n':  out->push_back('\n'); break;
                case 'r':  out->push_back('\r'); break;
                case 't':  out->push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    for (int i = 0; i < 4; ++i) {
                        if (p >= end) {
                            return fail("\\u escape is incomplete");
                        }
                        const int hv = hex_value(*p++);
                        if (hv < 0) {
                            return fail("\\u escape contains a non-hex character");
                        }
                        cp = (cp << 4) | (uint32_t) hv;
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    return fail(std::string("unknown escape: \\") + e);
            }
        }
        return fail("unterminated string");
    }

    bool parse_array(JsonValue* out) {
        ++p;  // '['
        out->type_ = JsonValue::Type::Array;
        skip_ws();
        if (p < end && *p == ']') {
            ++p;
            return true;
        }
        while (true) {
            JsonValue item;
            if (!parse_value(&item)) {
                return false;
            }
            out->array_.push_back(std::move(item));
            skip_ws();
            if (p >= end) {
                return fail("unterminated array");
            }
            if (*p == ',') {
                ++p;
                continue;
            }
            if (*p == ']') {
                ++p;
                return true;
            }
            return fail("missing , or ] after array element");
        }
    }

    bool parse_object(JsonValue* out) {
        ++p;  // '{'
        out->type_ = JsonValue::Type::Object;
        skip_ws();
        if (p < end && *p == '}') {
            ++p;
            return true;
        }
        while (true) {
            skip_ws();
            std::string key;
            if (!parse_string(&key)) {
                return false;
            }
            skip_ws();
            if (p >= end || *p != ':') {
                return fail("missing : after object key");
            }
            ++p;
            JsonValue value;
            if (!parse_value(&value)) {
                return false;
            }
            out->object_.emplace_back(std::move(key), std::move(value));
            skip_ws();
            if (p >= end) {
                return fail("unterminated object");
            }
            if (*p == ',') {
                ++p;
                continue;
            }
            if (*p == '}') {
                ++p;
                return true;
            }
            return fail("missing , or } after object member");
        }
    }
};

bool JsonValue::parse(const std::string& text, JsonValue* out, std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "output is null";
        }
        return false;
    }
    JsonParser parser;
    parser.p = text.c_str();
    parser.end = text.c_str() + text.size();
    JsonValue root;
    if (!parser.parse_value(&root)) {
        if (err != nullptr) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "JSON parse failed (offset %lld): %s",
                          (long long) (parser.p - text.c_str()), parser.err.c_str());
            *err = buf;
        }
        return false;
    }
    parser.skip_ws();
    if (parser.p != parser.end) {
        if (err != nullptr) {
            *err = "trailing content after the JSON value";
        }
        return false;
    }
    *out = std::move(root);
    return true;
}

bool JsonValue::as_bool(bool default_value) const {
    return type_ == Type::Bool ? boolean_ : default_value;
}

double JsonValue::as_number(double default_value) const {
    return type_ == Type::Number ? number_ : default_value;
}

int64_t JsonValue::as_int(int64_t default_value) const {
    return type_ == Type::Number ? (int64_t) number_ : default_value;
}

const std::string& JsonValue::as_string() const {
    return type_ == Type::String ? string_ : empty_string();
}

size_t JsonValue::size() const {
    if (type_ == Type::Array) {
        return array_.size();
    }
    if (type_ == Type::Object) {
        return object_.size();
    }
    return 0;
}

const JsonValue& JsonValue::at(size_t index) const {
    if (type_ != Type::Array || index >= array_.size()) {
        return null_value();
    }
    return array_[index];
}

const std::string& JsonValue::key_at(size_t index) const {
    if (type_ != Type::Object || index >= object_.size()) {
        return empty_string();
    }
    return object_[index].first;
}

const JsonValue& JsonValue::value_at(size_t index) const {
    if (type_ != Type::Object || index >= object_.size()) {
        return null_value();
    }
    return object_[index].second;
}

const JsonValue* JsonValue::find(const std::string& key) const {
    if (type_ != Type::Object) {
        return nullptr;
    }
    for (const auto& kv : object_) {
        if (kv.first == key) {
            return &kv.second;
        }
    }
    return nullptr;
}

bool json_read_file(const std::string& path, JsonValue* out, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        if (err != nullptr) {
            *err = "cannot open file: " + path;
        }
        return false;
    }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();
    return JsonValue::parse(text, out, err);
}

} // namespace common
} // namespace rvc_train
