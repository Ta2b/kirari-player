#pragma once

#include <filesystem>

namespace kirari::app::play {
struct PlayerPaths;
}

namespace kirari::app {

void add_backend_paths(play::PlayerPaths& paths, const std::filesystem::path& base);

}  // namespace kirari::app
