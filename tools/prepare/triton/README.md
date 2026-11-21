# Paquet Triton pour Kirari

Le module charge des noyaux précompilés. Le lecteur n'embarque ni Python ni PyTorch. Les paquets couvrent **SPAN Balanced et Performance HD v3.1 FP16 à 1920×1080**, avec ou sans Sharp1 selon les modèles installés, **DistilDRBA v2 lite FP16 RGB et mouvement à 1920×1088 et 1280×768**, et **RIFE 4.25 et 4.22 lite FP16 mouvement à 1920×1088**. Le lecteur vérifie l'empreinte ONNX et l'architecture du GPU. Les modèles ou formes absents du catalogue utilisent le chemin Vulkan du moteur. Changer les artefacts requiert de reconstruire le module avec leur nouvelle empreinte.

## Compiler le module

- NVIDIA Ada sous Linux ou Windows. Activer `KIRARI_WITH_TRITON=ON` et donner `KIRARI_TRITON_BUNDLE` au dossier de l'étape suivante. CMake utilise le SDK CUDA pour l'en-tête. À l'exécution il charge le pilote du système puis les CUBIN de l'architecture exacte, SM 8.9 par défaut. Renseigner `KIRARI_TRITON_CUDA_ARCH` pour un autre paquet CUDA produit avec le compilateur Triton sur la cible. Linux peut partager les tampons directement avec Vulkan avec un sémaphore timeline. Windows utilise les tampons de transfert hôte.
- AMD sous Linux ou Windows. Activer `KIRARI_WITH_TRITON=ON`, renseigner `KIRARI_TRITON_HIP_BUNDLE` et `KIRARI_TRITON_HIP_INCLUDE_DIR` si l'emplacement du SDK est différent de `HIP_PATH/include` ou de `/opt/rocm/include`. Le module charge le runtime HIP du système par son nom et sélectionne le GPU Vulkan par UUID ou adresse PCI. Sous Linux, le module importe les allocations de stockage Vulkan par FD lorsque le pilote et HIP le permettent. Le fil différé de Triton attend l'entrée, exécute le graphe dans ces tampons et signale la fin au lecteur. Windows et les imports non disponibles utilisent un tampon de transit. L'architecture du paquet se fixe avec `KIRARI_TRITON_HIP_ARCH` (`gfx1152` par défaut).
- Intel Arc A770 sous Linux ou Windows. Donner `KIRARI_TRITON_XPU_BUNDLE` et, au besoin, `KIRARI_TRITON_XPU_INCLUDE_DIR` au SDK d'en-têtes Level Zero. Le module charge le pilote Level Zero du système, sélectionne la carte par identité et lance le SPIR-V Kernel depuis ses tampons GPU. Cette voie utilise les transferts hôte. Le paquet vérifié hors ligne vise uniquement l'A770 (PCI 56a0).
- Les paquets NVIDIA, AMD et Intel peuvent cohabiter dans le même module. Chaque variable de paquet accepte une liste CMake. Par exemple, passer `"-DKIRARI_TRITON_BUNDLE=$balanced;$sharp1"` pour livrer les deux variantes sur NVIDIA. Le composant `triton` installe le premier paquet d'une architecture dans `triton/<architecture>`, puis les suivants dans `triton/<architecture>-<SHA-ONNX>`.

Pour comparer les modes de synchronisation sur NVIDIA/Linux, `KIRARI_TRITON_CUDA_TIMELINE=OFF` garde l'import mémoire et utilise les étapes différées du lecteur. `KIRARI_TRITON_CUDA_GRAPHS=ON` essaie de regrouper les lancements pour chaque paire de tampons réutilisée. Les attentes et signaux des sémaphores restent hors graphe.

## Préparer les fichiers

Créer le programme du réseau en suivant le [banc Triton de Kirari](../../bench/kernel_portable/triton_tune/README.md) avec Triton 3.8. À partir de son répertoire `paquet`, produire le plan et les empreintes utilisables par le module.

```sh
python tools/prepare/triton/paquet.py "$triton/paquet" "$destination_cuda"
```

Pour le paquet AMD, la compilation hors ligne requiert le chemin ROCm de Triton avec les mêmes tuiles et la même variante que le plan des poids. Par exemple, après génération du plan à 28 lancements et du jeu de poids avec `--body-tuned 128 4 2 --fuse-gates` :

```sh
python tools/bench/kernel_portable/triton_span/compile_amd.py gfx1152 "$hsaco" --body-tuned 128 4 2 --fuse-gates
python tools/prepare/triton/paquet.py "$triton/paquet" "$destination_hip" --hip "$hsaco"
```

Le plan s'appuie sur l'ABI des pointeurs des noyaux Triton, avec deux paramètres réservés mis à zéro. Le préparateur refuse un exécutable qui déclare de la mémoire de travail implicite. Il extrait la géométrie de bloc et la mémoire partagée propres à la cible AMD.

Pour le plan à 27 lancements, ajouter `--project-gate 64 2 2` à la fois à `network.py` pour le plan NVIDIA et à `compile_amd.py` pour le paquet AMD. La dernière porte et la projection s'exécutent dans le même noyau. Ce plan a été validé en lecture sur RTX 4070 Ti. Le paquet AMD a été compilé pour `gfx1152`.

Pour Performance et Sharp1 Performance, ajouter aussi `--compact-c48` aux mêmes deux commandes. Le plan à cinq blocs garde 18 lancements, avec les poids et activations stockés sur 48 canaux utiles. Le champ de grille existant donne X. Une ligne `kernel` peut porter Y et Z après le champ de mémoire partagée. Sans ces champs, Y et Z valent 1. Pour un paquet HIP ou XPU, le rapport de compilation utilise alors une liste `grille: [X, Y, Z]`.

Une ligne `view<TAB>nom<TAB>base<TAB>décalage<TAB>taille` crée une sous-vue d'une allocation interne déjà déclarée. Le décalage et la taille sont en octets. Les limites sont vérifiées à l'ouverture du plan et la vue partage la durée de vie de son tampon.

Le plan RGB DistilDRBA utilise `drba-triton-v1` et le SHA du modèle officiel dont les dimensions `H` et `W` ont été fixées par Kirari. Le plan déclare 23 noyaux et rejoue 45 lancements. Les entrées `pair01` et `pair12` ont chacune sept plans FP16 et la sortie `out` en a trois. Le plan lit seulement les plans RGB d'indices 3 à 5 de `pair12`. Sur la voie à tampons partagés et sémaphore GPU, Kirari prépare directement ces trois plans. Sélectionner Triton pour l'interpolation suffit pour charger le plan compatible.

Le plan de mouvement DistilDRBA utilise `drba-triton-motion-v1` et le SHA de la variante ONNX mouvement dont Kirari fixe ensuite les dimensions. Il reprend la même préparation d'entrée et produit un flot de quatre plans FP16 et un masque d'un plan. Le module prend en charge les deux tailles, choisies par la configuration du lecteur. Les deux tailles ont été chargées en lecture réelle sur RTX 4070 Ti. [Qualité et mesures](../../bench/kernel_portable/triton_distil/README.md)

L'option `--reemploi-encodage` produit un plan DistilDRBA à 24 noyaux qui sépare la mise à jour du temps des réductions et de l'encodage du trio. Avec des tampons partagés, le lecteur peut garder ces résultats pour d'autres instants du même trio.

Le plan RIFE utilise `rife-triton-motion-v1` et le SHA de la variante ONNX mouvement dérivée par Kirari. Il prend les onze plans FP16 du lecteur, dont la grille normalisée FP16, et rend le flot quatre plans et le masque un plan. Ses 29 ou 30 noyaux couvrent les deux encodages et les cinq IFBlocks. Les plans à 30 noyaux matérialisent le recalage D1 avant sa convolution en un tampon intégral ou un tampon réutilisé par bande. La soumission à tampons partagés peut reprendre un ou deux encodages sur indication du lecteur. [Qualité et mesures](../../bench/kernel_portable/triton_rife/README.md#chaîne-complète-rife-425-mouvement)

Le plan `rife-lite-triton-motion-v1` couvre RIFE 4.22 lite avec 25 noyaux et 71 lancements. Le flot porte le nom `/Add_5_output_0`. Le paquet emploie le même contrat d'entrée FP16 et le même réemploi des encodages. [Préparation et mesures](../../bench/kernel_portable/triton_rife/README.md#rife-422-lite)

Pour Intel Arc A770, utiliser **la roue Triton XPU 3.8** séparément de la roue standard, avec `spirv-val` et `spirv-dis`. Le générateur emploie un profil DG2 conservateur, sans API matérielle requise pendant la compilation. Il vérifie le SPIR-V 1.4, le point d'entrée, la géométrie et la signature de chaque noyau.

```sh
python tools/bench/kernel_portable/triton_span/compile_xpu.py "$spirv" --body-tuned 128 4 2 --fuse-gates
python tools/prepare/triton/paquet.py "$triton/paquet" "$destination_xpu" --xpu "$spirv"
```

## Limites de qualification

Les premiers tests concernent le chemin NVIDIA Ada sur Linux. La sortie `gfx1152` est compilée hors ligne. Une machine AMD et une machine Windows restent nécessaires pour contrôler chargement des noyaux, pixels et performances. La ROCm du GPU hôte doit exposer l'architecture du HSACO. Compiler un HSACO sous Linux ne démontre pas son chargement ni son exécution avec un pilote Windows donné.

Intel maintient son propre backend Triton qui produit des exécutables XPU. La liste de compatibilité couvre notamment Arc A770 et B580 et ne mentionne pas le Core Ultra 7 155H. Le support Windows est expérimental. Les six noyaux du réseau fusionné ont été émis et validés en SPIR-V Kernel pour le profil A770. Ce profil ne produit pas de DPAS dans les essais, sa vitesse doit être mesurée. Le chargeur Level Zero a été compilé avec les en-têtes officiels, l'exécution Intel reste à qualifier sur la carte. Les essais avec sous-groupe de huit lanes et multiplication matricielle ont échoué lors de l'abaissement dans le compilateur Intel. [Compatibilité Intel Triton](https://github.com/intel/intel-xpu-backend-for-triton#compatibility) · [Protocole Windows Intel](https://github.com/intel/intel-xpu-backend-for-triton/blob/main/.github/WINDOWS.md)
