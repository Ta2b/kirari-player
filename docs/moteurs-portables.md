# Moteurs de calcul portables

Le sélecteur **Paramètres → Vidéo → Général → Moteur de calcul** propose les moteurs dont le module et les ressources sont disponibles. Kirari conserve le choix dans `settings.toml`. Le journal identifie le moteur ouvert et chaque repli Vulkan.

**CUDA natif** et **TensorRT** sont deux choix du même module NVIDIA. CUDA natif active les chemins spécialisés pour les modèles qu'ils couvrent et utilise les plans TensorRT pour les autres réseaux. Le choix TensorRT dirige les réseaux vers les plans TensorRT et écarte les variantes natives de SPAN et les chemins hybrides de RIFE, DistilDRBA, GMFSS et TemporalFix. Les opérations de chaîne hors inférence (décodage, transferts, composition) partagent toujours les mêmes mécanismes. Le nom du moteur et le chemin SPAN sont consignés séparément pour les bancs.

Le moteur CUDA natif lance les noyaux compatibles dès l'ouverture du modèle choisi. RIFE, DistilDRBA et
GMFSS réessaient leur propre graphe sous TensorRT si la forme ou les poids ne sont pas couverts. Le lecteur
indique les refus et poursuit à la cadence source si l'interpolation sélectionnée ne peut pas démarrer.

Le SPAN CUDA spécialisé demande un module construit avec les noyaux CUTLASS. Le script de paquet Windows les construit avec `KIRARI_WITH_SPAN_FP8=ON`, qui inclut aussi la variante SPAN FP16.

## Préparer les modules

| Moteur | Configuration et ressources adjacentes au module |
|---|---|
| Vulkan | Livré avec le lecteur. Sous macOS, le lecteur passe par MoltenVK. |
| ZML | `KIRARI_WITH_ZML=ON`, puis `KIRARI_ZML_BUNDLE` et les en-têtes PJRT associés. Le chargeur choisit `zml/cuda`, `zml/rocm`, `zml/oneapi` ou `zml/metal` selon le GPU de rendu. [Préparer les greffons](zml.md) |
| HIP | `KIRARI_WITH_HIP=ON` et `KIRARI_HIP_ROCM_PREFIX` si le SDK se trouve ailleurs que sous `/opt/rocm`. CMake compile les binaires HSACO de `KIRARI_HIP_ARCHITECTURES` dans `hip/`. Le lecteur importe les tampons Vulkan sous Linux et les partage avec HIP, avec synchronisation par l'hôte. Sous Windows, la voie HIP passe par l'hôte et peut recevoir un HSACO précompilé via `KIRARI_HIP_CODE_DIRECTORY`. [Configuration HIP](../src/gpu/hip/CMakeLists.txt) |
| Slang | `KIRARI_WITH_SLANG=ON` et `KIRARI_SLANG_BUNDLE` vers un paquet précompilé. Linux et Windows utilisent des shaders Vulkan. macOS utilise un paquet de shaders Metal. [Module Slang](../src/gpu/slang/README.md) |
| Triton | `KIRARI_WITH_TRITON=ON` et un ou plusieurs paquets via `KIRARI_TRITON_BUNDLE` (NVIDIA), `KIRARI_TRITON_HIP_BUNDLE` (AMD) ou `KIRARI_TRITON_XPU_BUNDLE` (Intel) avec les en-têtes SDK. Le module cherche ses fichiers dans `triton/<architecture>/`. [Préparer et vérifier les paquets](../tools/prepare/triton/README.md) |

Un arbre de construction place les fichiers adjacents à l'exécutable. Pour installer les modules optionnels issus de ce même build, `cmake --install DOSSIER_BUILD --component hip --prefix DOSSIER_MODULES` copie le module HIP et ses ressources. Les composants `slang` et `triton` suivent le même principe. Le préfixe de la commande est le dossier contenant les modules du lecteur (`Contents/PlugIns` pour l'application macOS).

Les scripts de paquet Linux et Windows reprennent ces chemins depuis l'environnement. Une variable de paquet Slang ou Triton active le module correspondant. `KIRARI_WITH_HIP=ON` active HIP. Le script macOS inclut Slang et ZML si leur chemin de paquet est fourni.

**Release rapide** et **Release** intègrent les moteurs des paquets Linux et Windows depuis l'archive
épinglée dans [`portable-assets.toml`](../tools/release/portable-assets.toml). Les entrées
`moteurs_url` et `moteurs_sha256` permettent de choisir une autre archive pour un lancement.
Le script `tools/release/portable-assets.py` vérifie l'archive puis
les modules et catalogues assemblés. Pour ZML, la release Linux construit le module PJRT. Le greffon
du fabricant peut être installé dans le dossier de données de Kirari, y compris avec l'AppImage.
Windows reste sans greffon PJRT compatible avec la révision utilisée.

Pour qualifier le même source HIP sur NVIDIA, configurer aussi `KIRARI_HIP_CUDA_ARCHITECTURES=sm89` pour la RTX 4070 Ti. Le module lit `hip/span-sm89.cubin`, compilé par HIPCC ou fourni dans `KIRARI_HIP_CUDA_CODE_DIRECTORY`.

Exemple NVIDIA Ada sous Linux pour le développement, depuis un build Kirari déjà configuré. Les deux paquets pointent vers le dossier qui contient `plan.tsv` et `SHA256SUMS`.

```sh
cmake -S . -B "$build" \
  -DKIRARI_WITH_ZML=ON -DKIRARI_ZML_BUNDLE="$pjrt" \
  -DKIRARI_WITH_TRITON=ON -DKIRARI_TRITON_BUNDLE="$triton" \
  -DKIRARI_WITH_SLANG=ON -DKIRARI_SLANG_BUNDLE="$slang"
cmake --build "$build" --target kirari
```

Le paquet Slang qualifié couvre SPAN Balanced HD v3.1 FP16 à 1920×1080. Sur RTX, le catalogue Triton couvre aussi Sharp1 Balanced, Performance et Sharp1 Performance à cette résolution. Le lecteur choisit le paquet d'après l'empreinte ONNX et l'architecture GPU. HIP couvre cette géométrie avec les poids Balanced b8f64 préparés depuis l'ONNX. Les autres graphes et formes passent par Vulkan quand le module spécialisé refuse le format durant la préparation. La prise en charge d'un noyau matriciel dépend du modèle et des capacités du pilote. Le lecteur valide l'architecture GPU au moment d'ouvrir le module. Les transports par l'hôte de Slang, Triton ou ZML incluent les transferts dans le coût réel de la lecture.

Le test d'intégration sur RTX 4070 Ti sous Wayland a ouvert SPAN Balanced à 1080p avec les sept moteurs. Les journaux de CUDA et TensorRT identifient respectivement la variante spécialisée FP16 et le plan FP16. HIP a partagé l'entrée et gardé la sortie côté GPU sur les 86 passes de son test court. À 720p, Slang et Triton ont préparé le repli Vulkan et présenté des images.

Le [banc de lecture](../tools/bench/zml/lecture.py) accepte `--voies cuda tensorrt` avec un seul exécutable. Il exécute les parcours dans les deux ordres puis compare les images 4K des exports FFV1 10 bits. Les noms `hip`, `slang`, `triton`, `zml` et `vulkan` permettent d'étendre la comparaison.

Sur la RTX 4070 Ti sous Wayland, avec une source à 119,88 i/s, les deux passages de quatre secondes donnent au plus lent 98,25 images neuves/s en CUDA natif et 59,03 en TensorRT. Les p95 GPU de SPAN atteignent respectivement 11,05 et 17,90 ms. Le PSNR de la pire des 120 images 4K 10 bits alignées atteint 67,24 dB. [Rapport](../build/zml/mesures/portabilite-c64/integration-comparison.md)

Avec le même extrait et les mêmes paramètres, la série portable donne 74,58 images neuves/s en Triton, 34,32 en Vulkan et 26,87 en HIP sur NVIDIA. Les 120 images exportées par HIP et Triton sont identiques pixel par pixel. Le PSNR minimal face à Vulkan atteint 67,25 dB pour les deux. [Rapport](../build/zml/mesures/portabilite-c64/integration-portables.md)

## Potentiel par cible

Ce classement indique la voie à privilégier au banc de Kirari. Une compilation hors ligne établit l'ISA produite. Elle laisse à mesurer la qualité, le coût de partage des tampons et le débit sur l'appareil réel.

| GPU et système | Voie à privilégier | État du port |
|---|---|---|
| NVIDIA RTX, Linux | **CUDA natif** pour SPAN Balanced. **Triton** pour une implémentation matricielle partageable avec AMD. TensorRT reste le choix généraliste NVIDIA. | Sur RTX 4070 Ti, le microbanc Balanced donne 10,65–10,81 ms en CUDA direct. En composition, Triton fusionné a tenu 71,97 i/s, Slang Winograd 56,85 et ZML 41,35, au pire des deux passages. |
| NVIDIA RTX, Windows | **CUDA natif ou TensorRT** en premier. **Triton** comme alternative pour les noyaux SPAN. | Voie Triton et voie Slang compilables en principe avec les SDK Windows et les paquets de la carte. Les chemins de transfert hôte restent à vérifier sur Windows. |
| AMD RDNA3/3.5, Linux | Sur la Radeon 860M, **HIP** pour comparer en premier la voie WMMA FP16 avec tampons partagés. **Slang** est la voie portable à comparer sur le même pilote Vulkan. **Triton** peut aussi importer les tampons Vulkan via HIP sous Linux, avec synchronisation par l'hôte. Ce raccordement reste à valider sur la carte. | HSACO HIP et Triton compilés pour gfx1152, shaders Slang matriciels compilés hors ligne sur un profil RDNA3.5. ZML sait charger le greffon ROCm, qui doit couvrir le GPU et les opérations du modèle. |
| AMD, Windows | **Vulkan**, puis **Slang** si le pilote annonce les matrices coopératives requises. **Triton HIP** en essai sur un GPU couvert par le runtime AMD Windows. | Kirari HIP natif partage ses tampons sous Linux. Le chemin Windows utilise des transferts hôte et reste à qualifier. La Radeon 860M ne figure pas dans la liste officielle du HIP SDK Windows consultée. |
| Intel, Linux | **Vulkan** en base. **ZML oneAPI** pour tester des graphes entiers sur les GPU couverts. Slang est un candidat si les matrices coopératives requises sont exposées. | Le paquet Triton-XPU A770 contient les six noyaux validés hors ligne. Cette variante utilise le calcul scalaire. L’exécution Level Zero reste à vérifier. La liste Intel ne comprend pas le Core Ultra 7 155H. |
| Intel, Windows | **Vulkan**, puis **Slang** sur un pilote et un GPU qui rendent les matrices requises. | Le backend Intel Triton Windows est expérimental en amont. La voie Level Zero de Kirari pour Arc A770 reste à construire et à tester sous Windows. |
| Apple Silicon, macOS | **ZML Metal** pour la couverture de graphes et **Slang Metal** pour mesurer le potentiel de SPAN spécialisé. Core ML reste la voie système disponible avec son module. | Le paquet MSL est généré depuis le source portable. Le build Objective-C++ et la lecture sur M4 Pro restent à qualifier. |

Les moteurs NVIDIA ne sont pas une voie native sur les Mac Apple Silicon. Les Mac Intel/Radeon anciens restent hors de la cible Metal de ces paquets. Les greffons ZML fournis par le préparateur couvrent CUDA, ROCm et oneAPI sous Linux et Metal sous macOS arm64. Le chargeur Windows est prévu pour un greffon PJRT DLL compatible. La distribution ZML vérifiée ne fournit pas ce paquet Windows natif.

[Cibles ZML](https://github.com/zml/zml) · [Cibles Slang](https://shader-slang.org/slang/user-guide/targets) · [Triton Windows](https://github.com/triton-lang/triton-windows) · [Triton Intel](https://github.com/intel/intel-xpu-backend-for-triton#compatibility) · [AMD HIP SDK Windows](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/shared/hipsdk/reference/system-requirements.html)
