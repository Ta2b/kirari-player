#pragma once

#include <filesystem>
#include <sstream>

#include "gpu/ir/ir.h"
#include "gpu/ir/shape.h"

namespace kirari::gpu::zml {
using ir::Dims;

struct Value {
  std::string id;
  Dims dims;
  std::string element;
};
std::string array(const Dims& values);
Dims axes(std::size_t count, std::int64_t start = 0);

// Émet le dialecte StableHLO typé. Les formes sont fixées avant l'émission.
class Hlo {
 public:
  void enter_node(std::string_view name);
  Value op(std::string_view name, const std::vector<Value>& args, const Dims& dims, std::string_view element,
           std::string_view attrs = {});
  Value reduce(const Value& operand, const Dims& reduced, bool maximum = false);
  Value scatter_nd(const Value& data, const Value& indices, const Value& updates);
  Value constant(const ir::Value& data);
  Value number(double value, std::string_view element);
  Value broadcast(const Value& operand, const Dims& dims, const Dims& mapping);
  Value expand(const Value& operand, const Dims& dims);
  Value convert(const Value& operand, std::string_view element);
  Value reshape(const Value& operand, const Dims& dims);
  Value transpose(const Value& operand, const Dims& permutation);
  Value binary(std::string_view op, const Value& lhs, const Value& rhs, const Dims& dims);
  Value slice(const Value& operand, const Dims& start, const Dims& end, const Dims& strides);
  std::string module(const std::vector<Value>& inputs, const std::vector<Value>& outputs) const;

 private:
  std::ostringstream body_;
  std::string location_;
  unsigned next_ = 0;
};

std::expected<std::string, Refusal> lower_onnx(const std::filesystem::path& path,
                                               std::span<const ir::InputShape> inputs,
                                               std::string_view io_element);
std::expected<std::vector<Value>, Refusal> operation(Hlo& hlo, const ir::Graph& graph, const ir::Node& node,
                                                     const std::vector<Value>& values);
std::expected<Value, Refusal> attention(Hlo& hlo, const ir::Graph& graph, const ir::Node& node,
                                        const std::vector<Value>& values);
std::expected<Value, Refusal> sample(Hlo& hlo, const ir::Graph& graph, const ir::Node& node,
                                     const std::vector<Value>& values);
std::string_view element_type(ir::DataType type);
}  // namespace kirari::gpu::zml
