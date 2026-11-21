#pragma once
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace c64 {
namespace fs = std::filesystem;
using U = std::uint32_t;
using Bytes = std::vector<std::byte>;

inline void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
inline Bytes read(const fs::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  require(in && in.tellg() > 0, "fichier attendu");
  Bytes bytes(static_cast<std::size_t>(in.tellg()));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  require(in.good(), "lecture incomplète");
  return bytes;
}
inline U dimension(std::string_view arg, U limit, U minimum = 1) {
  U n = 0;
  const auto* end = arg.data() + arg.size();
  const auto parsed = std::from_chars(arg.data(), end, n);
  require(parsed.ec == std::errc{} && parsed.ptr == end && n >= minimum && n <= limit, "entier hors limites");
  return n;
}

}  // namespace c64
