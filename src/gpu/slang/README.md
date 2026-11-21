# Moteur Slang

Les modules Slang exécutent SPAN Balanced FP16 1920×1080 → 3840×2160. Le paquet SPIR-V
Winograd transforme les 24 convolutions du corps. Avec `--fuse-gates`, chaque troisième convolution
et sa porte résiduelle partagent un dispatch. Le plan passe de 36 à 28 lancements et conserve
l'arrondi FP16 intermédiaire. Les plans antérieurs à portes séparées restent utilisables.

- Sous Linux, le module SPIR-V sélectionne le même `VkDevice` physique par UUID. Il importe les
  tampons et le sémaphore Vulkan du lecteur par descripteurs. Les entrées et sorties restent sur le GPU.
- Sous Windows, le module SPIR-V conserve les activations intermédiaires sur le GPU. Les deux extrémités
  transitent par des tampons hôte cohérents à chaque image. La construction exige la bibliothèque
  d'importation `vulkan-1` du SDK Vulkan. Le chemin de données par l'hôte a produit deux sorties 4K
  identiques au banc résident lors du test Vulkan sur RTX/Linux. La DLL reste à construire et tester sous Windows.
- Sous macOS, le module Objective-C++ charge les sept shaders MSL en matrices SIMD 8×8 sur Apple
  Silicon, avec les deux extrémités en mémoire partagée et des copies hôte. Ce paquet direct est
  distinct du paquet SPIR-V Winograd. L'émission Slang des sept sources et les indices de leurs
  paramètres ont été contrôlés. La compilation du module et des shaders par Apple et la lecture
  composée demandent encore un poste macOS.

`KIRARI_WITH_SLANG=ON` active la cible. `KIRARI_SLANG_BUNDLE` nomme un paquet du modèle
précompilé. Le manifeste `SHA256SUMS` lie le plan, les poids et les SPIR-V à l'identité de
l'exécutable. Il sert également à séparer les caches lors d'un changement de paquet.

Depuis la racine du dépôt, avec les données du banc et le compilateur Slang :

```bash
essais="$PWD/build/zml/mesures/portabilite-c64"
export LD_LIBRARY_PATH="$PWD/build/rd-slang/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 tools/bench/kernel_portable/slang_tuning/package.py \
    "$essais/slang-span/paquet" "$essais/triton-span/paquet" "$essais/slang-tuning/reseau-winograd-portes" \
    --mode winograd --fuse-gates --slangc "$PWD/build/rd-slang/usr/bin/slangc"
cmake --preset linux-release -DKIRARI_WITH_SLANG=ON \
    -DKIRARI_SLANG_BUNDLE="$essais/slang-tuning/reseau-winograd-portes"
cmake --build --preset linux-release --target kirari
```

`cmake --install` inclut le composant `slang`, qui installe le module et le dossier `slang/`
adjacent. Les autres modèles et dimensions sont servis par le moteur Vulkan du module.

En SPIR-V, le périphérique doit exposer les matrices coopératives 16×16×16 à opérandes et
accumulateur FP16, ainsi que des sous-groupes de 32. L'extension de contrôle de taille des
sous-groupes permet de demander 32 sur les pilotes offrant 64 par défaut. Sur Linux, le partage
exige en plus les imports par descripteur.

Générer le paquet Metal depuis les mêmes données avant de configurer l'option Slang sur macOS.

```bash
python3 tools/bench/kernel_portable/slang_tuning/package_metal.py     "$essais/slang-span/paquet" "$essais/triton-span/paquet" "$essais/slang-tuning/reseau-metal"     --slangc "$PWD/build/rd-slang/usr/bin/slangc"
# Sur macOS, fournir le dossier réseau-metal à KIRARI_SLANG_BUNDLE.
```

## Resize expérimental pour l’interpolation

Avec `KIRARI_SLANG_RESIZE4=ON` et les outils `slangc`, `spirv-opt` et `spirv-val`, le module Vulkan de Slang peut substituer `slang_resize4_f16` aux Resize bilinéaires FP16 ×4 à coordonnées demi-pixel. Le shader emploie l’arène Vulkan du graphe. Le partage des sommes horizontales conserve les deux arrondis verticaux.

Sur la RTX 4070 Ti, le banc `kirari_slang_resize4` retrouve les codes de la référence CPU sans FMA pour les huit cas NCHW et NHWC16. Le shader Vulkan de référence diffère sur 25 codes au total. Le profil en lecture confirme deux dispatchs Slang dans DistilDRBA v2 lite et un dans RIFE 4.22 lite, au sein de leurs graphes Vulkan. La substitution demeure optionnelle pendant la qualification sur des extraits complets.
