// RVC-Train-GGML - minimal JSON parser (src/common/json.h)
//
// Purpose: read golden manifest.json (since M2-2) and future training configs (M7).
// No third-party dependencies: supports only the JSON standard subset (object/array/string/number/bool/null),
// string escapes support \" \\ \/ \b \f \n \r \t \uXXXX (BMP, surrogate pairs not combined).
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rvc_train {
namespace common {

class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    static bool parse(const std::string& text, JsonValue* out, std::string* err);

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool        as_bool(bool default_value = false) const;
    double      as_number(double default_value = 0.0) const;
    int64_t     as_int(int64_t default_value = 0) const;
    const std::string& as_string() const;

    size_t size() const;                             // Array/Object element count, 0 otherwise
    const JsonValue& at(size_t index) const;         // Array index (returns Null out of range)
    const std::string& key_at(size_t index) const;   // Object key at index (returns empty string out of range)
    const JsonValue& value_at(size_t index) const;   // Object value at index (returns Null out of range)
    const JsonValue* find(const std::string& key) const;  // Object lookup (nullptr when not found)

private:
    friend struct JsonParser;

    Type        type_ = Type::Null;
    bool        boolean_ = false;
    double      number_ = 0.0;
    std::string string_;
    std::vector<JsonValue> array_;
    std::vector<std::pair<std::string, JsonValue>> object_;
};

// Read a whole file and parse it (returns false and writes err on failure)
bool json_read_file(const std::string& path, JsonValue* out, std::string* err);

} // namespace common
} // namespace rvc_train
