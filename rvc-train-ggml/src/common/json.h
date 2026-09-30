// RVC-Train-GGML - 极简 JSON 解析（src/common/json.h）
//
// 用途：读取 golden manifest.json（M2-2 起）与后续训练配置（M7）。
// 不引入第三方依赖：只支持 JSON 标准子集（对象/数组/字符串/数字/布尔/null），
// 字符串转义支持 \" \\ \/ \b \f \n \r \t \uXXXX（BMP，代理对不合并）。
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

    size_t size() const;                             // Array/Object 元素数，其他返回 0
    const JsonValue& at(size_t index) const;         // Array 下标（越界返回 Null）
    const std::string& key_at(size_t index) const;   // Object 第 index 个键（越界返回空串）
    const JsonValue& value_at(size_t index) const;   // Object 第 index 个值（越界返回 Null）
    const JsonValue* find(const std::string& key) const;  // Object 查找（未找到 nullptr）

private:
    friend struct JsonParser;

    Type        type_ = Type::Null;
    bool        boolean_ = false;
    double      number_ = 0.0;
    std::string string_;
    std::vector<JsonValue> array_;
    std::vector<std::pair<std::string, JsonValue>> object_;
};

// 读取整个文件并解析（失败返回 false 并写入 err）
bool json_read_file(const std::string& path, JsonValue* out, std::string* err);

} // namespace common
} // namespace rvc_train
