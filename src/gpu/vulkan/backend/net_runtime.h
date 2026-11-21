// Réseaux confiés à un autre runtime (Core ML sur Mac) derrière le backend Vulkan : la chaîne (ingestion,
// rife_input, warp, ring…) garde les noyaux Vulkan sur l'appareil de rendu, seul le réseau passe par le
// runtime du fabricant.
//
// Les tenseurs des réseaux vivent en mémoire hôte visible du GPU (MemoryUse::Readback, en cache
// côté hôte) : le runtime les lit et les écrit par leur projection, après l'attente sur l'hôte
// de la file Vulkan (BackendCaps::needs_host_sync), et rend la main une fois ses sorties écrites.
// Sur un GPU intégré, cette mémoire hôte peut être celle du runtime (Runtime::allocate_host),
// importée par Vulkan : le réseau n'y copie plus rien.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/rife_features.h"
#include "gpu/vulkan/compute/caps.h"

namespace kirari::gpu::vk::net {

// Un tenseur d'entrée ou de sortie : NCHW dense, dans le type du réseau, vu par l'hôte (data).
struct HostBinding {
  std::byte* data = nullptr;
  std::uint64_t bytes = 0;
  TensorType type = TensorType::F16;
  Shape shape;
  std::uintptr_t device = 0;  // adresse du même tenseur dans le runtime, quand la mémoire GPU est partagée
};

// Mémoire hôte allouée par le runtime, que la chaîne Vulkan importe (VK_EXT_external_memory_host) :
// sur un GPU intégré (Core ML, IOSurface), le réseau lit et écrit les tenseurs de la chaîne
// sans aucune copie. Rendue à sa destruction, après le tampon Vulkan qui la voit.
class HostMemory {
 public:
  virtual ~HostMemory() = default;
  virtual std::byte* data() const = 0;
};

// La forme fixée d'une entrée, par son nom.
struct InputShape {
  std::string name;
  std::vector<std::int64_t> dims;
};

// Un réseau chargé par le runtime.
class RuntimeNet;
class RuntimeSpanRegions {
 public:
  virtual ~RuntimeSpanRegions() = default;
  virtual kernels::SpanTileGrid grid() const = 0;
  virtual Shape previous_shape() const = 0;
  virtual std::vector<std::uint16_t> plan(std::span<const std::uint8_t> core) const = 0;
  virtual RuntimeNet& network() = 0;
};

class RuntimeRifeFeatures {
 public:
  virtual ~RuntimeRifeFeatures() = default;
  virtual void forget() = 0;
  virtual std::string summary() const = 0;
  virtual std::expected<std::shared_ptr<void>, Refusal> run_device(RifeFeatureRequest,
                                                                   std::span<const HostBinding>,
                                                                   std::span<const HostBinding>,
                                                                   std::uint64_t, std::uint64_t) = 0;
};

class RuntimeTimeReuse {
 public:
  virtual ~RuntimeTimeReuse() = default;
  virtual void forget() = 0;
  virtual std::string summary() const = 0;
  virtual std::expected<std::shared_ptr<void>, Refusal> run_device(bool same_images,
                                                                   std::span<const HostBinding>,
                                                                   std::span<const HostBinding>,
                                                                   std::uint64_t, std::uint64_t) = 0;
};

class RuntimeNet {
 public:
  // Présent seulement si le réseau régional a déjà été préparé et désérialisé sur cet appareil.
  std::unique_ptr<RuntimeSpanRegions> span_regions;
  bool pair12_rgb_only = false;  // propriété du programme effectivement chargé
  std::unique_ptr<RuntimeRifeFeatures> rife_features;
  std::unique_ptr<RuntimeTimeReuse> time_reuse;
  virtual ~RuntimeNet() = default;
  virtual std::span<const std::string> input_names() const = 0;
  virtual std::span<const std::string> output_names() const = 0;
  // Calcule les sorties, écrites avant le retour. Refus : celui du runtime, nommé.
  virtual std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                           std::span<const HostBinding> outputs) = 0;
  // Les entrées sont disponibles à wait. La copie des sorties signale done sur la timeline importée.
  // La file garde le prêt rendu jusqu'à ce signal, y compris les références PJRT.
  virtual std::expected<std::shared_ptr<void>, Refusal> run_device(std::span<const HostBinding>,
                                                                   std::span<const HostBinding>,
                                                                   std::uint64_t, std::uint64_t) {
    return std::unexpected(Refusal{RefusalCode::FeatureUnavailable, "timeline du runtime indisponible"});
  }
  // Comme run, sans bloquer l'appelant : sur son propre fil, le runtime attend ready() (les
  // entrées écrites : une attente hôte de la file Vulkan, hors du fil qui nourrit le GPU),
  // calcule, puis appelle done(résultat) une fois ses sorties écrites (le signal hôte de l'étape
  // sur la timeline, que la soumission suivante attend côté GPU). L'appelant ne touche ni aux
  // entrées ni aux sorties avant done ; ready et done viennent d'un autre fil, et les appels se
  // suivent dans l'ordre. Refus FeatureUnavailable (par défaut) avant de rien lancer :
  // l'appelant fait alors l'étape par run.
  // Les rappels se passent par valeur : les implémentations les gardent jusqu'à la fin du travail.
  // NOLINTBEGIN(performance-unnecessary-value-param)
  virtual std::expected<void, Refusal> run_deferred(
      std::span<const HostBinding> /*inputs*/, std::span<const HostBinding> /*outputs*/,
      std::function<std::expected<void, Refusal>()> /*ready*/,
      std::function<void(std::expected<void, Refusal>)> /*done*/) {
    return std::unexpected(
        Refusal{RefusalCode::FeatureUnavailable, "exécution différée non prise en charge"});
  }
  // NOLINTEND(performance-unnecessary-value-param)
};

struct DeviceMemory {
  std::uintptr_t address = 0;
  std::shared_ptr<const void> owner;
};

struct IoNames {
  std::vector<std::string> inputs, outputs;
};

class Runtime {
 public:
  virtual ~Runtime() = default;
  virtual std::string_view name() const = 0;  // BackendCaps::name : « migraphx », « openvino »
  virtual std::string device_name() const = 0;
  virtual std::string version() const = 0;  // entre dans la clé des artefacts
  // Ce que build fait du modèle, quand cela change sans que version change : entre aussi dans la
  // clé, et une autre recette reconstruit les artefacts. Vide : aucune (la clé d'avant).
  virtual std::string_view recipe() const { return {}; }
  virtual std::string_view extension() const = 0;
  // Compile le modèle, ses entrées fixées à ces formes (vide : celles du fichier, toutes
  // statiques), et l'écrit à path. Refus : ceux du runtime, Cancelled.
  virtual std::expected<void, Refusal> build(const ModelSpec& model, std::span<const InputShape> inputs,
                                             const std::filesystem::path& path, Progress& progress) = 0;
  // Les runtimes à compilation en deux temps écrivent un exécutable voisin de leur source.
  // nullopt : build suffit. false : la préparation doit terminer ou compléter un ancien cache.
  virtual std::optional<bool> compiled_artifact_ready(const std::filesystem::path&) const { return {}; }
  virtual std::expected<void, Refusal> prepare_compiled_artifact(const std::filesystem::path&, const IoNames&,
                                                                 Progress&) {
    return {};
  }
  virtual std::expected<std::unique_ptr<RuntimeNet>, Refusal> load(const std::filesystem::path& path) = 0;

  virtual std::expected<void, Refusal> import_timeline(int) {
    return std::unexpected(Refusal{RefusalCode::FeatureUnavailable, "import de timeline indisponible"});
  }
  virtual bool shares_device_memory() const { return false; }
  // Importe une allocation Vulkan dédiée, décalage 0. Le descripteur reste la propriété de Vulkan.
  virtual std::expected<DeviceMemory, Refusal> import_device(std::uintptr_t, std::size_t, std::size_t) {
    return std::unexpected(Refusal{RefusalCode::FeatureUnavailable, "import de mémoire GPU indisponible"});
  }

  // Mémoire hôte du runtime pour un tenseur (GPU intégré) : bytes octets, adresse multiple de
  // alignment, que le backend importe dans Vulkan ; type et shape, ceux du tenseur qu'elle portera
  // (Core ML en fait une IOSurface à sa forme). Refus FeatureUnavailable par défaut : le backend
  // prend alors sa propre mémoire hôte, que le runtime copie, pour tous les tenseurs ;
  // FormatUnsupported : pour ce tenseur seulement.
  virtual std::expected<std::unique_ptr<HostMemory>, Refusal> allocate_host(std::uint64_t /*bytes*/,
                                                                            std::uint64_t /*alignment*/,
                                                                            TensorType /*type*/,
                                                                            const Shape& /*shape*/) {
    return std::unexpected(
        Refusal{RefusalCode::FeatureUnavailable, "mémoire hôte du runtime non prise en charge"});
  }
};

// Crée le runtime sur l'appareil de cette description Vulkan (même UUID, sinon le premier du
// fabricant). Refus DeviceMismatch : appareil d'un autre fabricant, que le backend Vulkan seul
// sert ; LibraryUnavailable, DeviceUnavailable : runtime absent ou sans appareil.
using RuntimeFactory = std::function<std::expected<std::shared_ptr<Runtime>, Refusal>(const DeviceCaps&)>;

// Noms des entrées et des sorties d'un réseau, rangés à côté de son artefact (un par ligne,
// « in » ou « out » puis le nom), pour les runtimes dont l'artefact ne les garde pas.
std::expected<void, Refusal> write_io_names(const std::filesystem::path& artifact,
                                            std::span<const std::string> inputs,
                                            std::span<const std::string> outputs);
std::expected<IoNames, Refusal> read_io_names(const std::filesystem::path& artifact);

// Commun à la préparation et au chargement direct d'une ancienne source. Hors de la famille
// publiée/supprimée par publish_artifact et discard_artifact.
std::filesystem::path artifact_lock_path(const std::filesystem::path& artifact);

// Publie sous final_path l'artefact construit à staged, avec ses fichiers voisins (même nom
// suivi d'une extension : noms d'E/S, appareil choisi) : chacun par renommage, les noms d'E/S
// en dernier, qui font foi pour read_io_names. Un arrêt ou un second processus ne laisse donc
// jamais un artefact tronqué sous le nom final. Refus CacheIo.
std::expected<void, Refusal> publish_artifact(const std::filesystem::path& staged,
                                              const std::filesystem::path& final_path);

// Efface l'artefact et ses fichiers voisins, pour que la construction suivante le refasse.
// Ce qui ne s'efface pas reste ; l'appelant a déjà son refus à rendre.
void discard_artifact(const std::filesystem::path& artifact);

// Copie count éléments de from vers to en convertissant entre F32 et F16 (même type : une copie).
// Refus FormatUnsupported : un type entier différent de l'autre.
std::expected<void, Refusal> copy_converted(const std::byte* from, TensorType from_type, std::byte* to,
                                            TensorType to_type, std::size_t count);

// Les noms de l'interface d'un fichier ONNX, lus par gpu_ir.
std::expected<IoNames, Refusal> onnx_io_names(const std::filesystem::path& onnx,
                                              std::vector<InputShape>* declared = nullptr);

// Les dimensions symboliques des entrées d'un fichier ONNX fixées par ces formes : (symbole,
// valeur), une fois chacun. Refus InputMismatch : un symbole qui recevrait deux valeurs.
using DimOverride = std::pair<std::string, std::int64_t>;
std::expected<std::vector<DimOverride>, Refusal> symbolic_overrides(const std::filesystem::path& onnx,
                                                                    std::span<const InputShape> inputs);

// Le fichier ONNX contient-il un nœud de ce type (domaine par défaut) ? Refus : lecture, décodage.
std::expected<bool, Refusal> onnx_uses_op(const std::filesystem::path& onnx, std::string_view op_type);

}  // namespace kirari::gpu::vk::net
