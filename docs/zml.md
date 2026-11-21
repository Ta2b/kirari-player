# Moteur ZML

ZML se sélectionne dans l’écran des moteurs ou dans **Vidéo → Général** quand le greffon de la carte est installé. Il ouvre les modèles ONNX du lecteur et garde ses modèles compilés dans le cache de la carte. Le décodage, les traitements de l’image, la composition et l’export utilisent la chaîne Vulkan, via MoltenVK sur Mac. Les réseaux passent par le greffon PJRT de ZML.

| Plateforme amont | Cartes ciblées | Vérification dans Kirari |
|---|---|---|
| CUDA sous Linux | NVIDIA | Lecture, export et contrôle numérique sur RTX 4070 Ti |
| Metal sous macOS arm64 | Apple Silicon | Binaire du greffon arm64 vérifié. Compilation et lecture sur Mac à vérifier |
| oneAPI sous Linux x86-64 | Intel | Chargement et couverture des opérations à vérifier sur carte |
| ROCm sous Linux x86-64 | AMD pris en charge par le runtime amont | Chargement et couverture des opérations à vérifier sur carte |

La révision amont épinglée distribue aussi CUDA sous Linux arm64. La disponibilité d’un greffon dépend du système, du modèle de GPU et de son pilote. Le paquet Windows peut recevoir un greffon PJRT Windows compatible. La révision ZML utilisée ici ne fournit pas de tel binaire. [Plateformes de ZML](https://github.com/zml/zml/tree/f430c3530fbc6d99f734b641eb8f3fd699868843/platforms)

## Construire

Installer les dépendances habituelles du lecteur pour la plateforme et Bazelisk (ou Bazel 9.3.0). Le script prend la cible et un dossier local. Il vérifie l’empreinte des en-têtes OpenXLA et assemble les bibliothèques et outils d’exécution du fabricant. Le script construit également le frontal Zig pour la préparation des modèles. La préparation Metal ajoute le compilateur et ses ressources. Chaque appel ajoute une plateforme dans le même dossier.

```sh
tools/deps/zml-runtime.sh metal "$PWD/build/deps-zml"
export KIRARI_ZML_BUNDLE="$PWD/build/deps-zml/runtimes"
tools/release/package-macos.sh "$PWD/build/paquet-zml"
```

Sous Linux, remplacer `metal` par `cuda`, `rocm` ou `oneapi` et employer le script de paquet Linux. Lors d’une première configuration CMake, `KIRARI_ZML_BUNDLE` active le moteur et permet de trouver les en-têtes dans le dossier `include` voisin de `runtimes`. Sur un build déjà configuré avec ZML désactivé, passer aussi `-DKIRARI_WITH_ZML=ON`. Le chemin du greffon permet de lancer le banc réseau directement depuis le build.

```sh
cmake --preset macos-release \
  -DKIRARI_ZML_BUNDLE="$KIRARI_ZML_BUNDLE" \
  -DKIRARI_ZML_METAL_PLUGIN="$KIRARI_ZML_BUNDLE/metal/lib/libzml_metal.dylib"
cmake --build --preset macos-release --target kirari kirari_zml_check
```

Les autres chemins de développement sont `KIRARI_ZML_CUDA_PLUGIN`, `KIRARI_ZML_ROCM_PLUGIN` et `KIRARI_ZML_ONEAPI_PLUGIN`. Sur NVIDIA, le cache de l’exécutable PJRT s’ajoute à celui du module StableHLO. Sous Linux, le lecteur partage ses tampons Vulkan en mémoire GPU locale avec PJRT CUDA. Les entrées deviennent des vues PJRT et les sorties sont copiées directement vers les tampons GPU du lecteur. Quand le pilote accepte un sémaphore timeline externe, les deux files se transmettent la propriété des tampons par ce sémaphore. Les autres plateformes disposent du chemin portable en mémoire accessible à l’hôte. Le journal enregistre la proportion de passes avec partage GPU.

Sur le greffon CUDA sérialisable, la préparation compile le StableHLO, vérifie le nombre de sorties et écrit l’exécutable PJRT avant de déclarer le moteur prêt. Le chargement suivant réutilise cet exécutable. Une compilation interne de cuDNN déclenchée par la toute première inférence reste visible dans la durée de lecture.

Le binaire CMake trouve à côté de lui les modules activés et le frontal Zig, avec ses fichiers d’exécution. Sous Linux NVIDIA, le propriétaire du poste prépare les bibliothèques CUDA de ZML localement. Après avoir configuré le build avec le bundle issu de `tools/deps/zml-runtime.sh cuda`, il peut aussi les installer à côté d’un lecteur distribué équipé du module ZML.

```sh
cmake --install build/linux-release --component zml_cuda_local --prefix /chemin/du/dossier/de/kirari
```

Le composant de publication `zml` contient le module Kirari. Quand le frontal ou les runtimes
ROCm, oneAPI ou Metal sont fournis au build, il embarque aussi ces ressources. Le composant local CUDA
a sa propre étape d’installation. Sous Linux, y compris avec l'AppImage, le lecteur cherche également les
greffons et le frontal sous `${XDG_DATA_HOME:-$HOME/.local/share}/kirari/zml`. Le dossier `runtimes`
produit par le préparateur contient directement l'arborescence attendue.

Sous Linux, ces commandes construisent le lecteur avec le même agencement de dossiers.

```sh
cmake --preset linux-release -DKIRARI_ZML_BUNDLE="$KIRARI_ZML_BUNDLE"
cmake --build --preset linux-release --target kirari
```

## Ce qui est compilé

Kirari importe les poids et la topologie ONNX. Le frontal C++ spécialisé émet du StableHLO. Le plugin fourni par ZML compile ce module avec XLA pour l’appareil choisi. Les noyaux CUDA de Kirari restent dans le moteur natif. Le frontal du dossier `src/gpu/zml/sdk`, écrit en Zig avec le SDK officiel, construit des `zml.Tensor` et produit lui aussi du StableHLO. La provenance du module et sa résolution font partie des données à conserver à chaque comparaison. [Modèles du SDK ZML](https://docs.zml.ai/tutorials/write_first_model/)

Le GPU peut exécuter des fusions générées par XLA ou Triton et des opérations que XLA confie à une bibliothèque du fabricant. L’absence de `custom_call` dans le module source suffit seulement à exclure un appel inscrit par le frontal. Le HLO optimisé permet notamment de reconnaître les fusions cuDNN dont l’opcode XLA est `fusion`. Le script `tools/bench/zml/origine.py` donne le détail. [Génération GPU d’OpenXLA](https://openxla.org/xla/gpu_architecture)

## Ordre et isolation des mesures

1. Figer les exécutables, les modules et les poids. Utiliser un profil propre et la même source pour les deux moteurs. Le test réseau vérifie d’abord les sorties avant tout chronométrage.
2. Examiner le StableHLO et le HLO optimisé. Pour SPAN, compter les convolutions et repérer l’allocation de la concaténation. Pour RIFE et DistilDRBA, vérifier les formes réellement échantillonnées, les têtes et les encodeurs.
3. En lecture réelle, chronométrer successivement ZML et le témoin natif, puis les mêmes voies en ordre inverse. Relever les images neuves, les images manquées et le p95 GPU du rôle. Pour RIFE, distinguer la sortie de mouvement suivie du warp du lecteur et le graphe image complet (`--rife-image`).
4. Une fois les chronos terminés, effectuer les deux exports FFV1, vérifier le format 10 bits et tous les PTS, puis calculer le PSNR de la pire image. Les tests réseau et les comparaisons de pixels sur CPU ne mesurent pas la vitesse de lecture.

Les scripts de lecture prennent le même verrou interprocessus dans les deux worktrees. `python3 tools/bench/zml/serie.py commande …` permet de soumettre un contrôle numérique, un test unitaire ou un profileur à cette file. Une compilation JIT et son autotuning font partie du même essai. Les builds CMake ou Bazel qui utilisent uniquement le CPU peuvent se préparer à priorité basse dans un répertoire indépendant. Le banc multi-modèles libère le verrou après chaque préchauffage et chaque lecture, tout en conservant l’ordre alterné de ses deux moteurs. Il le reprend pour la comparaison des exports de chaque modèle. L’export des captures Nsight en SQLite et leur analyse hors ligne peuvent se faire à part.

### Programme d’optimisation sur la RTX 4070 Ti

| Ordre | Point de départ sur le poste | Expérience qui décide de la suite |
|---|---|---|
| 1 | Après fusion des huit portes de Balanced, 25 lancements du principal groupe de convolutions occupent 14,07 ms par passe. | Tester un produit matriciel implicite sur les véritables pixels intérieurs de C1. Contrôler les écritures temporaires avant d'ajouter le traitement des bords. |
| 2 | Les huit portes combinent le produit 1×1 et tout son épilogue sur les unités matricielles. Les 18 concaténations de la projection finale occupent encore 0,83 ms par passe. | Fusionner les quatre sources de la projection directement dans le produit matriciel. Vérifier la vitesse du réseau entier, la lecture et les 120 images FFV1. |
| 3 | Les blocs RIFE lisent une fraction des coordonnées de warp. Le calcul au centre D2 gagne en 720p et 1080p sur 4.22 lite. | Reconnaître la topologie de chaque autre modèle avant de porter le traitement. Contrôler les plans de convolution, les sorties mouvement/masque, les échéances manquées et les 185 images exportées en ×8. |
| 4 | DistilDRBA rejoint le débit natif en 1080p ×8. Le p95 du rôle reste à 5,73 ms. Sur la trace, le warp final prend 1,62 ms et la copie de sortie 0,031 ms au p95. | Repartir du warp ou des calculs répétés sur les mêmes images. Utiliser le même programme PJRT pour isoler un changement de transport. Les deux ordres de passage en lecture décident de l’activation. |
| 5 | Le couplage des deux réseaux GMFSS fonctionne sur GPU. Les corrélations peuvent amplifier une petite erreur sur les pixels d’une paire. | Vérifier les entrées capturées dans le lecteur contre l’ONNX, puis comparer deux compilations à froid sur les mêmes 95 images. |

Le banc `modeles.py --kirari-zml-temoin CHEMIN --zml-seul --passes 2 --prechauffer` compare deux constructions ZML dans l’ordre témoin, candidat, candidat, témoin. Chacune garde son cache. Le rapport donne aussi le nombre d’images, le nombre de mesures du rôle et la cadence sur la durée totale de la session. Ces compteurs rendent apparent un amorçage qui prendrait plusieurs secondes.

Le banc `modeles.py --zml-seul --reference DOSSIER` relit l’extrait FFV1 de la référence et calcule les pixels du seul candidat. Une référence absente laisse la mesure de qualité ouverte. Pour collecter une trace, utiliser `--trace-zml --zml-seul --passes=1` avec un seul modèle, puis exporter la capture en SQLite avec Nsight Systems. Les durées de la trace servent au diagnostic ; les mesures publiées utilisent une lecture sans instrumentation.

## Vérifier un appareil

Le banc réseau compare chaque sortie du vrai modèle à l’interpréteur sur une image déterministe. Une dimension de 128 convient aux modèles d’interpolation du catalogue. Le dossier de sortie doit être neuf. Le banc de lecture lance deux passages par moteur dans des ordres inversés, exporte le même extrait en FFV1, vérifie les PTS et compare chaque image. Utiliser une source 1080p à 59,94 i/s pour comparer à la RTX. Une source lente borne la cadence mesurable.

```sh
build/macos-release/src/gpu/zml/kirari_zml_check "$PWD/build/controle-reseau" 128 "$MODELE"
python3 tools/bench/zml/lecture.py --kirari "$LECTEUR" --model "$MODELE" \
  --video "$VIDEO" --expected-gpu "M4 Pro" --out "$PWD/build/lecture-m4"
```

`MODELE` désigne le SPAN AnimeJaNai HD V3.1 Balanced b8f64 FP16 pour le banc de lecture. `--kirari-cuda` ajoute une comparaison avec le lecteur équipé du module CUDA natif sur NVIDIA. Le journal de ZML doit confirmer l’exécution du réseau. Les chiffres « images neuves soumises » restent exploitables quand le compositeur ne date qu’une partie des présentations.

## Couverture mesurée sur RTX 4070 Ti

- SPAN AnimeJaNai Balanced et Performance, avec et sans Sharp1, ainsi que les deux ArtCNN ont passé le contrôle numérique en FP16.
- RIFE 4.18, 4.22 lite, 4.25, 4.25 lite et 4.26, ainsi que DistilDRBA, ont passé le contrôle numérique. Le lecteur a exécuté RIFE 4.22 lite avec SPAN ainsi que DistilDRBA en interpolation ×2.
- Les trois modèles TemporalFix du catalogue passent par le frontal Zig de ZML. En entrée 128 × 128, les sorties des modèles S1, S2 et S3 présentent un écart RMS de 0,000119 à 0,000142 face à l’interpréteur ONNX de Kirari. Les deux graphes dérivés pour le cache de S2 passent aussi ce contrôle. En lecture réelle à 640 × 360, TemporalFix Normal encode 105 images sur GPU et traite 102 fenêtres de sept images, toutes en partage GPU. Les deux frontaux, Zig et StableHLO C++, soumettent 97 images en quatre secondes. Les 48 images FFV1 10 bits aux mêmes horodatages atteignent au minimum 74,79 dB entre ces deux voies ZML.
- Les graphes de paire et de fusion de GMFSS Base et Union ont passé le contrôle numérique. Le lecteur a exécuté Union à 342 × 192 avec RIFE. Avec `--motion=gmfss`, une lecture séparée du panoramique a exécuté GMFSS à chaque instant, avec 103/103 entrées GPU partagées pour le réseau de paire et 102/102 pour la fusion. Elle a soumis 191 images. Les seuls réseaux exécutés dans cette lecture sont ceux de GMFSS. La préparation des images, le paramètre temporel et le remplissage intermédiaire sont calculés sur GPU. Les deux réseaux utilisent PJRT. Le banc Vulkan de cette étape compare notamment les collisions de projection et les pixels sans contribution à la référence CPU. Les corrélations des modèles à normalisation de couche conservent leurs intermédiaires en FP32.

### Interpolation en lecture réelle

Deux passages de 6 s par voie et par modèle, en ordre inversé, sur RTX 4070 Ti. Source animée 23,976 i/s, interpolation ×2 à 1280 × 720, fenêtre Wayland. Les binaires et les poids sont figés pour toute la série. Chacun des douze exports fournit 47 images FFV1 10 bits aux mêmes PTS que son témoin. Les comparaisons portent ici sur le frontal C++ StableHLO. Le graphe de mouvement de RIFE fournit le flot et le masque, puis le lecteur mesure séparément le warp final.

| Modèle | ZML, p95 du rôle | NVIDIA, p95 du rôle | PSNR minimal |
|---|---:|---:|---:|
| RIFE 4.18 | 19,47 ms | 22,89 ms | 79,65 dB |
| RIFE 4.22 lite | 16,58 ms | 8,41 ms | 72,47 dB |
| RIFE 4.25 | 16,29 ms | 6,12 ms | 72,46 dB |
| RIFE 4.25 lite | 11,51 ms | 4,48 ms | 66,65 dB |
| RIFE 4.26 | 17,85 ms | 5,92 ms | 70,48 dB |
| DistilDRBA v2 lite | 68,80 ms | 1,77 ms | 77,33 dB |

Les deux voies soumettent environ 33,2 images neuves/s sur cet extrait avec ses dessins tenus. Pour RIFE 4.18, TensorRT exécute le modèle de mouvement entier après refus de la découpe fusionnée. ZML gagne sur chacun des deux passages. Les quatre autres RIFE et DistilDRBA disposent de leur découpe native fusionnée. Pour DistilDRBA, le rôle ne compte que neuf appels chronométrés par passage. Son p95 est donc le maximum observé. Les médianes de ZML sont de 5,84 et 5,24 ms, contre 1,13 et 0,82 ms pour le natif. Un extrait à mouvement plus continu est nécessaire pour préciser le p95.

### Frontal Zig officiel en lecture

La révision ZML `f430c3530fbc6d99f734b641eb8f3fd699868843` trace les réseaux avec `zml.Tensor`. Le lecteur compile le module avec le greffon PJRT CUDA sur la RTX 4070 Ti. `modeles.py --exiger-sdk` vérifie la provenance et le journal contrôle le partage GPU à chaque passe. Les références NVIDIA sont les exports originaux. Le minimum du PSNR porte sur les mêmes 47 images FFV1 pour le mouvement et les mêmes 180 images pour le SPAN.

| Modèle et formulation Zig | Pire p95 du rôle | Images neuves/s | PSNR minimal contre CUDA |
|---|---:|---:|---:|
| RIFE 4.22 lite, réglage témoin | 20,57 ms | 33,20 | 72,56 dB |
| RIFE 4.22 lite, agrandissements bilinéaires par phases | 16,93 ms | 33,20 | 72,55 dB |
| RIFE 4.22 lite, phases et warp final à indices linéaires | 17,78 ms | 33,20 | 72,49 dB |
| RIFE 4.25, réglage témoin | 15,51 ms | 33,20 | 72,29 dB |
| RIFE 4.25, agrandissements bilinéaires par phases | 17,71 ms | 33,20 | 72,17 dB |
| DistilDRBA, témoin sur l’extrait d’animation | 66,46 ms | 33,19 | 77,32 dB |
| DistilDRBA, réductions directes sur l’extrait d’animation | 66,99 ms | 33,03 | 77,32 dB |
| SPAN Balanced, témoin 1080p → 4K à 59,94 i/s | 32,84 ms | 34,36 | 64,43 dB |
| SPAN Balanced, huit portes 1×1 exprimées en produits matriciels | 30,92 ms | 34,12 | 64,43 dB |
| SPAN Balanced, projection finale 1×1 exprimée en un seul produit sur les quatre entrées | 31,60 ms | 33,76 | 64,43 dB |

Les quatre modèles SPAN utilisent les poids FP16 du catalogue. Pour qualifier une mesure du SDK, `modeles.py --exiger-sdk` croise le nom du réseau dans le journal d’interopérabilité et le nom du fichier MLIR chargé. Il vérifie la provenance Zig et les appels explicites dans ce fichier. Les fusions générées et les appels à cuDNN décidés pendant la compilation se lisent séparément dans l’exécutable PJRT. Un ancien banc local produisait environ 60 i/s sur Balanced en appelant `kirari.span64.vertical.v1` depuis son MLIR. Ce chiffre appartient à l’expérience avec noyau CUDA Kirari.

Les témoins de mouvement de ce tableau utilisent la convolution transposée par phases. La ligne « agrandissements bilinéaires par phases » modifie uniquement le redimensionnement. Sur 4.22 lite, la capture du chemin StableHLO C++ pour cette transformation retrouve 54 passes identiques. Les 115 lancements détaillés initiaux deviennent 108 noyaux détaillés et deux graphes CUDA agrégés. En réunissant les deux types d’activité sans chevauchement, le temps GPU occupé médian passe de 7,922 à 6,209 ms. Les copies finales prennent 0,016 puis 0,015 ms. Une capture avec Nsight Compute situe les lectures et écritures de la concaténation initiale à 73,32 et 52,41 Mo par appel. La capture et la mesure de lecture se font séparément.

Le panoramique dense à 23,976 i/s permet également de comparer les cinq RIFE en lecture continue ×2. La fenêtre mesure 1600 × 900 et le travail 1280 × 720, avec deux passages de dix secondes par moteur, caches préchauffés. Cette série utilise le frontal Zig officiel, la version antérieure à la réutilisation des commandes Vulkan et le lecteur CUDA du fil des noyaux natifs figé pour les cinq modèles. La référence emploie le réseau TensorRT entier sur 4.18, dont la découpe native est refusée dès le bloc 0. Les quatre autres références emploient la découpe du lecteur. Les 24 images source de chaque extrait FFV1 sont identiques entre moteurs. Les 23 images intermédiaires déterminent le minimum sur 47.

| RIFE | ZML / NVIDIA, images neuves par seconde | Manquées ZML / NVIDIA | p95 ZML / NVIDIA | PSNR minimal |
|---|---:|---:|---:|---:|
| 4.18 | 47,89 / 48,00 | 0,21 % / 0 % | 13,37 / 16,48 ms | 77,11 dB |
| 4.22 lite | 47,82 / 48,00 | 0,42 % / 0 % | 12,06 / 5,89 ms | 55,61 dB |
| 4.25 | 47,81 / 47,99 | 0,42 % / 0 % | 12,36 / 6,10 ms | 56,82 dB |
| 4.25 lite | 47,81 / 48,01 | 0,42 % / 0 % | 9,74 / 5,51 ms | **49,44 dB** |
| 4.26 | 47,80 / 47,99 | 0,42 % / 0 % | 13,08 / 5,78 ms | 56,65 dB |

RIFE 4.25 lite est sous le seuil de 50 dB face à la voie native découpée sur ce panoramique. Le troisième export, obtenu avec `tools/bench/zml/temoin_rife.py`, conserve le réseau de mouvement ONNX entier dans TensorRT. Les deux voies TensorRT emploient le même warp CUDA externe. Sur les 47 images, ZML atteint au moins 69,68 dB face à ce témoin entier et la voie découpée 49,41 dB. Sur l’interpolée à 522 ms, les valeurs sont respectivement de 72,51 et 49,41 dB. Les deux images source adjacentes sont identiques dans les trois exports. Les grilles FP16 des warps internes de l’ONNX et les coordonnées calculées en pixels dans la décomposition diffèrent. La contribution de cette différence à l’écart mesuré reste ouverte.

Sur l’extrait d’animation à 23,976 i/s du banc CUDA, les deux lecteurs ont ensuite traité les cinq modèles en interpolation ×8 à 720p. La sortie Wayland était réglée à 165 Hz, sans agrandissement ni restauration. Chaque modèle a eu deux lectures de six secondes par moteur, caches préchauffés, et son propre export de 185 images FFV1 10 bits aux mêmes dates. La cadence soumise tient compte des dessins tenus. Les retours datés du compositeur couvrent 366 à 389 images sur les 483 à 506 soumises par passage.

| RIFE | Images neuves/s ZML / CUDA | Sources manquées ZML / CUDA | Rôle p95 ZML / CUDA | PSNR minimal |
|---|---:|---:|---:|---:|
| 4.18 | 82,59 / 80,41 | 5,52 / 8,52 % | 8,35 / 10,30 ms | 77,57 dB |
| 4.22 lite | 83,90 / 83,72 | 6,15 / 4,91 % | 7,33 / 3,48 ms | 71,69 dB |
| 4.25 | 83,73 / 83,70 | 5,09 / 4,91 % | 7,37 / 3,24 ms | 70,50 dB |
| 4.25 lite | 84,04 / 83,88 | 4,90 / 5,07 % | 6,03 / 2,88 ms | 59,63 dB |
| 4.26 | 83,74 / 83,58 | 4,91 / 5,10 % | 7,47 / 3,17 ms | 67,37 dB |

Le cache expérimental de RIFE 4.25 peut conserver sur le GPU les traits des deux images et les relire aux instants suivants. Sur l’extrait d’animation à 720p ×8, il a servi 240 reprises d’une paire, 10 transitions réutilisant une image et 47 paires complètes. Les 297 appels ont partagé leurs tampons GPU. Sur deux lectures par voie, le pire p95 passe de 7,89 à 8,02 ms et la cadence de 83,68 à 83,55 images neuves/s. Les 185 images exportées sont identiques avec et sans cache et atteignent 70,64 dB minimum face au CUDA archivé. Le cache reste désactivé dans les réglages livrés.

Le lecteur borne l’estimation du prochain rendu à une période d’écran quand il dispose d’une seule mesure. Cela évite de sauter des images déjà prêtes après un premier rendu inhabituellement long. Sur un passage RIFE, le témoin saute de 0 à 62,5 ms de temps vidéo après un rendu de 108,4 ms. Dans le passage candidat avec premier rendu de 92,6 ms, les dates de 21 et 42 ms sont présentées. Quand le premier rendu prend environ 4 ms, les deux versions présentent les mêmes premières dates et rencontrent le même trou à 62,5 ms. Sur DistilDRBA, le premier rendu prend 3,6 à 3,9 ms dans les quatre passages et la première paire attend ensuite 60,5 à 82,2 ms. Ce réglage de présentation cible le premier cas. Il est indépendant du temps de calcul annoncé pour PJRT.

Une comparaison RIFE 4.25 a inclus les convolutions classiques dans les graphes de commandes XLA avec `--xla_gpu_enable_command_buffer=+CONVOLUTION`. Dans le greffon livré, la création du handle cuDNN échoue si la première convolution du client a lieu durant la capture. L’exécution préalable d’une convolution StableHLO sur le même client rend la capture possible. NVIDIA documente le chargement à la demande des bibliothèques et noyaux lors du premier appel. [Premier appel cuDNN](https://docs.nvidia.com/cudnn/latest/reference/faq/) · [Graphes de commandes OpenXLA](https://github.com/openxla/xla/blob/fc31477d352ed8edc223648524382f9b10a7699b/xla/backends/gpu/runtime/command_buffer_thunk.cc)

Sur une paire dense résidente, chaque série de 100 inférences est précédée de 15 appels et les deux bras sont initialisés de la même façon. Les médianes de soumission CPU passent de 0,703–0,729 ms à 0,023 ms, et le temps jusqu’à disponibilité des deux sorties passe de 3,096–3,345 ms à 2,941–2,946 ms. Flot et masque sont identiques bit pour bit. Sur le lecteur, une petite convolution de préparation de 16×16 est exécutée dans les deux bras. Les 59 plans cuDNN du RIFE sont identiques entre programmes. Le pire p95 en lecture passe de 10,89 à 10,32 ms sur deux passages par bras, avec 481 images soumises, aucune manquée et 47 images FFV1 identiques. Les séries se recouvrent en latence. Ces chiffres qualifient le lecteur expérimental sur cette carte. Un essai avec les trois appels `cudnnGraphVersionCheck`, `cudnnOpsVersionCheck` et `cudnnCnnVersionCheck` réussit les contrôles de version en 0,51 ms, puis rencontre de nouveau l’erreur cuDNN 4004 si la première exécution est capturée. L’initialisation effective du client est déterminante.

La convolution de préparation seule a été comparée au lecteur normal avec exactement le même programme PJRT. Sur deux lectures de six secondes par bras, les deux sorties vidéo sont identiques sur 47 images pour RIFE 4.25 et DistilDRBA. Sur RIFE, chaque bras manque jusqu’à 0,69 % des images et le pire p95 passe de 8,82 à 9,28 ms. Sur DistilDRBA, les manquées passent de 1,04 à 0,35 % et le p95 de 5,02 à 5,20 ms. Le temps du lancement à la première présentation varie de 1,04 à 1,22 s sur la série. Ce préchauffage reste au banc.

Pour le DistilDRBA Zig, les réductions directes lisent les quatre pixels qui contribuent au filtre bilinéaire. Dans le HLO compilé à 1280 × 768, six intermédiaires passent de 12,66 à 2,64 Mio cumulés. Un panoramique synthétique à 23,976 i/s sollicite le réseau environ 240 fois en dix secondes. Sur deux séries en ordres inversés, le pire p95 de chaque série vaut 8,80 puis 8,40 ms pour le témoin et 7,36 puis 8,40 ms pour le candidat. Cette scène a son propre protocole. Les chiffres CUDA du tableau précédent concernent l’extrait d’animation.

Sur SPAN, la capture du frontal Zig donne 15,10 ms pour les 24 convolutions 3×3 du corps, 3,47 ms pour les huit convolutions d’attention et 5,28 ms pour les sept épilogues séparés. Les noyaux Tensor Core cuDNN dominants lisent du FP16 et accumulent en FP32. Sur cette combinaison de greffon et de GPU, demander globalement les sommes FP16 laisse une convolution avec biais sans plan cuDNN. La demande explicite `ALG_DOT_F16_F16_F16` pour les seules portes est également refusée par le compilateur. Le candidat matriciel mesuré laisse XLA choisir l’accumulation et reste expérimental.

Sur RTX 4070 Ti, le lecteur active désormais les huit portes fusionnées pour les poids Balanced FP16 du catalogue en 1920×1080. Chaque région du frontal Zig comprend le produit matriciel 1×1, le biais, tanh, l'addition résiduelle et la multiplication finale. Le lecteur inscrit temporairement une transformation via l'extension C officielle PJRT XlaTransform. Elle sélectionne l'émetteur `__triton_gemm` pour ces huit régions identifiées. XLA choisit les tuiles et produit les deux symboles machine utilisés, avec 32 instructions statiques `HMMA.16816.F32` par symbole. Un greffon sans l'extension, un refus durant la préparation, une autre carte ou d'autres poids suivent le graphe dense. La configuration CMake `KIRARI_ZML_SPAN_BALANCED_TRITON_GATES=OFF` redonne le témoin sur cette carte. [Extension PJRT XlaTransform](https://github.com/openxla/xla/blob/fc31477d352ed8edc223648524382f9b10a7699b/xla/pjrt/c/pjrt_c_api_xla_transform_extension.h) · [Émetteurs GPU de XLA](https://openxla.org/xla/emitters)

| SPAN Balanced, source à 119,88 i/s, sans réemploi | Images neuves soumises/s | Images/s avec retour de présentation | Manquées | SPAN p95 |
|---|---:|---:|---:|---:|
| ZML dense, deux passages alternés | 34,65 | 34,50 | 12,97 % | 32,63 ms |
| ZML huit portes, deux passages alternés | 41,43 | 41,38 | 17,70 % | 25,27 ms |
| ZML huit portes, lecteur Release préparé depuis un cache vide | 41,53 | 41,49 | 14,18 % | 25,28 ms |
| CUDA natif du dixième lot, séance de référence | 96,60 | 96,69 | 19,28 % | 11,05 ms |

Chaque valeur est celle du pire des deux passages pour l'indicateur. Dans le lecteur Release, Wayland date 825 images sur 939 puis 812 sur 926. Le film de 8,992 s demande 22,27 à 22,57 s sur l'hôte. Les 120 images 4K FFV1, alignées par PTS, donnent au minimum 68,66 dB face à CUDA. Le banc alterné donne 78,94 dB entre le nouveau ZML et le précédent. Le pont C++ et le candidat qui servait à isoler l'émetteur produisent exactement les mêmes 24 883 200 valeurs sur la capture résidente. Les 27 convolutions restantes du pont y utilisent les mêmes plans que le dense. Aucun appel de noyau Kirari n'apparaît dans les programmes sources.

Avec le même lecteur Release et les mêmes réglages sur l'écran Wayland 165 Hz, quatre lectures alternées du film à 119,88 i/s donnent 27,79 / 41,66 / 41,41 / 27,76 images neuves/s en Vulkan / ZML / ZML / Vulkan. Sur le pire passage de chaque moteur, ZML soumet ainsi 49,2 % d'images de plus, avec un p95 de 25,37 ms contre 37,71 ms. La cadence sur la partie datée est de 41,44 et 27,77 i/s. Les manquées atteignent respectivement 15,48 % et 11,68 % des PTS parcourus.

Sur le même graphe à huit portes, Nsight Systems retrouve 91 lancements dans chacun des 249 intervalles de sortie après amorçage. Les 25 noyaux du principal groupe de convolutions occupent 14,07 ms et les huit portes 5,63 ms en moyenne. La projection par bandes lance 18 concaténations totalisant 0,83 ms. La copie de sortie GPU→GPU prend 0,23 ms. Le temps libre entre les noyaux de ce contexte CUDA est de 0,24 ms en médiane et 0,64 ms au p95.

Une expérience sur la première convolution du bloc Balanced a permis d’identifier la précision réellement exécutée lorsque le biais et la ReLU restent dans un noyau séparé. Les deux graphes compilés contiennent alors un seul `__cudnn$convForward`. Avec la demande FP16, Nsight Compute retrouve 32 instructions statiques `HMMA.16816.F16` dans le noyau CUTLASS choisi par cuDNN. Avec la demande FP32, Nsight Systems observe un noyau XMMA de calcul FP32. Sur deux séries de 100 inférences, les médianes de la couche entière avec son épilogue s’établissent à 0,808–0,809 ms et 0,894 ms respectivement. L’erreur L2 relative entre ces deux sorties vaut 0,00126. L’option `--xla_disable_hlo_passes=cudnn-fused-convolution-rewriter` s’applique aux deux mesures. La formulation de production, où cuDNN peut fusionner la ReLU, avait donné environ 0,60 ms sur cette couche.

Une sonde indépendante, `tools/bench/zml/graphe_cudnn_span.py`, isole le choix du plan pour cette même première convolution 3×3. Elle utilise les poids de SPAN, un biais FP16 et un épilogue biais/ReLU calculé en FP32. Sur les plans que l’heuristique B de cuDNN a fournis, 15 en FP16 et 14 en FP32, les meilleurs confirmés prennent respectivement 0,372 et 0,505 ms. Nsight Compute a retrouvé 32 instructions statiques `HMMA.16816.F16` dans le premier noyau et 16 `HMMA.16816.F32` dans le second. L’erreur L2 relative de la sortie du plan FP16 vaut 0,00139 face à la référence PJRT FP32 de la couche. Ces plans ont été mesurés dans la sonde cuDNN. Seul un exécutable produit par le compilateur puis qualifié sur le film pourra établir un gain pour Kirari. [Graphes d’opérations cuDNN](https://docs.nvidia.com/cudnn/overview)

Le SDK épingle l’archive PJRT CUDA `202611201220.397.1.2b5a0ee63c3b`. ZML relie cette série à son dépôt `xla-private`. Pour étudier le choix des types nous avons examiné les sources publiques OpenXLA de la révision `fc31477` référencée par le SDK. Ces sources associent au même réglage le type d’accumulation et celui de l’activation. Le profilage porte sur le binaire distribué. [Archive choisie par ZML](https://github.com/zml/zml/blob/f430c3530fbc6d99f734b641eb8f3fd699868843/platforms/cuda/cuda.bzl#L230-L237) · [Publication de ZML](https://github.com/zml/zml/commit/8fcff8c462070d06056f8a85ba0c8a81f0f3a636) · [Abaissement de la convolution fusionnée](https://github.com/openxla/xla/blob/fc31477d352ed8edc223648524382f9b10a7699b/xla/stream_executor/cuda/cuda_dnn.cc#L3506-L3520)

Dans un essai antérieur aux régions explicites, la concaténation FP16 de forme `1×540×960×256` restait une fusion séparée. Cette formulation matérialisait 253,125 Mio par image sur Balanced. Le lecteur a ensuite employé une projection en bandes de 30 lignes.

Sur les deux SPAN Performance C48, le module SDK peut aussi fusionner les cinq portes. La série alternée témoin, candidat, candidat, témoin donne au pire 66,27 et 67,03 images neuves/s sur Performance et 66,21 et 66,90 sur Sharp1 Performance. Le p95 passe de 15,70 à 15,79 ms pour le premier et de 15,78 à 15,66 ms pour le second. Sur une qualification indépendante de 120 images FFV1 pour chacun, les minima contre CUDA sont respectivement de 70,34 et 70,02 dB. Pour les deux réseaux complets, le total des allocations temporaires de l’exécutable PJRT passe de 298 604 544 à 248 836 864 octets. Le lecteur active les cinq portes sur ces deux modèles en 1080p sur RTX 4070 Ti. L’option `KIRARI_ZML_SPAN_PERFORMANCE_TRITON_GATES=OFF` rétablit le témoin.

### Choix automatiques pour RIFE 4.22 lite et 4.25

Sur PJRT CUDA, le SDK reconnaît la topologie FP16 de 4.22 lite et active le calcul par phases des convolutions transposées et des agrandissements bilinéaires. Sur deux lectures de neuf secondes avec préchauffage séparé, ajouter les phases bilinéaires au témoin à convolutions transposées par phases ramène le pire p95 de 13,13 à 11,72 ms. Le module du lecteur de production confirme six agrandissements et quatre IFBlocks par phases sur 4.22 lite. Le choix de RIFE 4.25 est indépendant. `KIRARI_ZML_AUTO_RIFE422_PHASES=OFF` permet de reproduire le témoin avec les deux transformations automatiques désactivées.

Sur deux passages de six secondes avec le lecteur de production et le même extrait, les onze plans originaux donnent 17,17 ms sur 4.22 lite et 16,78 ms sur 4.25. Générer les quatre plans de maillage dans le graphe à partir de constantes compactes donne respectivement 19,33 et 16,94 ms. Les quatre exports dépassent 72 dB sur les 47 images alignées. La voie normale conserve les onze plans. Les versions RIFE du banc et du catalogue ont les mêmes poids FP16 pour ces deux modèles. Leurs exports ONNX diffèrent par des conversions explicites.

Pour RIFE 4.25 FP16 sur le SDK CUDA, le réglage automatique applique les phases aux convolutions transposées de l’encodeur et des cinq IFBlocks. Il conserve les Resize d’origine. Avec préchauffage séparé, deux passages de neuf secondes donnent 15,52 ms contre 16,30 ms pour l’ONNX du banc et 16,45 ms contre 18,21 ms pour l’ONNX distribué. Sur le modèle distribué lancé depuis un profil neuf, le premier p95 de six secondes monte à 19,02 ms contre 15,85 ms. Le second passage redescend à 15,15 ms contre 15,63 ms. `KIRARI_ZML_AUTO_RIFE425_TRANSPOSE_PHASES=OFF` permet de reproduire le témoin.

L'audit hors ligne `tools/bench/zml/beta_rife.py` retrouve dans les exécutables des modèles distribués chaque mot FP16 des filtres et biais après multiplication par bêta, sur les 32 ResConv de 4.22 lite et les 40 de 4.25. L'entrée résiduelle est le quatrième opérande de la même convolution cuDNN. La LeakyReLU suit l'appel. Une préparation séparée de ces mêmes poids dans Kirari dupliquerait le travail déjà fait par le compilateur.

Sur la version mouvement de 4.22 lite, une permutation des 14 plans RVB et caractéristiques du dernier bloc, avec permutation inverse des poids correspondants, donne 14,84 ms contre 17,17 ms au pire p95 et 72,40 dB sur les 47 images. Les images neuves baissent légèrement et le temps GPU occupé médian sous Nsight passe de 6,96 à 7,27 ms. Cette permutation reste au banc.

Un autre essai de 4.22 lite compacte les coordonnées XY du warp. Le sous-graphe passe de 33,75 à 24,375 Mio de données de géométrie, mais l’allocation temporaire globale du programme reste à 139 607 304 octets. La lecture donne 19,25 ms au pire p95 contre 17,17 ms au témoin, avec 72,55 dB sur l’export. La variante reste au banc.

L’essai distinct `KIRARI_ZML_FOLD_RESCONV` ajoute aussi l’identité au centre diagonal du filtre et présente le LeakyReLU sous forme de sélection. Avec la fusion à l’exécution activée, l’audit des exécutables retrouve respectivement 32 et 40 appels cuDNN à trois opérandes avec la bonne pente. Les exports de 47 images atteignent 72,45 et 72,19 dB. En revanche, la première inférence d’une nouvelle instance dure environ 11,7 s sur 4.22 lite et 6,7 s sur 4.25, même avec le programme PJRT en cache. Les lectures de six secondes qui en résultent mesurent principalement cet amorçage. Ce réglage reste désactivé. [Moteurs de fusion cuDNN](https://docs.nvidia.com/cudnn/latest/developer/graph-api/)

Sur le **RIFE 4.25 dense du catalogue**, cette même fusion enlève les 40 noyaux d’activation et conserve 59 convolutions. Le banc résident fixe les 19 plans dont le descripteur reste commun. Avec deux images du film en entrée GPU et 100 inférences chaudes par série, les médianes de l’ordre témoin, fusion, fusion, témoin valent **3,083 / 3,116 / 3,033 / 3,075 ms**. Le premier calcul dans ce client prend 92 ms pour le témoin et 3,87 s pour la fusion. En lecture composée, les quatre passages soumettent les mêmes 415 images entre 10 et 22 s, dont 127 interpolées, à 34,59–34,60 images/s. Le délai initial pénalise la lecture entière de 24 s, où le pire taux de manquées passe de 0,12 à 17,65 %.

Sur les exécutables sérialisés de cette lecture, Nsight attribue 3,979 des 3,990 s d’émission hôte du premier calcul fusionné aux 40 convolutions résiduelles, dont 3,026 s dans les huit du bloc 3. Les 59 appels `cudnnBackendExecute` totalisent 7,91 ms sur l’hôte. Le programme conserve les choix d’algorithme et de configuration des convolutions. XLA reconstruit les plans d’exécution cuDNN au premier usage. L’option de fusion des résiduels reste désactivée. [Banc résident](../tools/bench/zml/temps_paire_gmflow/main.zig) · [Reconstruction du plan dans OpenXLA](https://github.com/openxla/xla/blob/fc31477d352ed8edc223648524382f9b10a7699b/xla/stream_executor/cuda/cuda_dnn.cc#L5638-L5670)

Le fragment D2 de RIFE 4.22 lite regroupe deux warps, les redimensionnements et la première convolution du bloc. Un portage ZML réduit les neuf canaux de masque et de caractéristiques directement aux centres de la demi-résolution. Les dix-neuf autres canaux gardent leur calcul. Sur les entrées synthétiques résidentes du microbanc et les poids réels, la médiane des deux séries est de 1,021 et 1,009 ms en direct et de 0,702 et 0,689 ms pour cette variante. Le temporaire passe de 64 391 952 à 59 477 256 octets. L’erreur L2 relative de l’activation est de 0,000207. Pour 4.25 les médianes observées sont plus dispersées (1,147 et 0,649 ms en direct, 0,942 et 0,545 ms pour la variante). Les deux modèles donnent un écart absolu maximal de 0,001953.

En lecture avec composition, deux passages de neuf secondes après préchauffage font passer le pire p95 de 4.22 lite de 16,13 à 14,84 ms sur le modèle de banc et de 15,76 à 14,70 ms sur le modèle distribué. Les deux modèles ont les mêmes poids mais des Cast différents. Les deux réglages soumettent 313 puis 314 images sur le modèle distribué. Sur le graphe de banc, le minimum est de 72,50 dB face à CUDA sur 47 images. Le même essai sur 4.25 donne 14,86 ms en direct et 16,82 ms avec D2, à 72,35 dB. L’export du modèle 4.22 lite distribué atteint 72,02 dB au pire sur 47 images face à CUDA. Le lecteur active D2 sur le SDK CUDA pour sa topologie FP16 à onze plans en 1280×768. `KIRARI_ZML_AUTO_RIFE422_D2_CENTRE=OFF` redonne le témoin. Le réglage de lecture de 4.25 conserve son redimensionnement.

Le lecteur active aussi D2 sur le RIFE 4.22 lite distribué en 1920×1088, la taille rembourrée d’une lecture 1080p. En ×8 sur l’animation à 23,976 i/s et en fenêtre Wayland à 165 Hz, la première série alternée passe de 137,30 à 139,42 images neuves/s et de 16,92 à 15,42 % d’images manquées. La seconde, en ordre inverse, passe de 138,99 à 140,92 images/s et de 14,67 à 12,82 % manquées. Le p95 du rôle complet reste proche de 11 ms. CUDA natif atteint 163,60 images/s et 2,92 ms. Les 48 plans cuDNN sont identiques entre les variantes ZML. Les 185 images FFV1 comparées aux mêmes dates donnent 94,50 dB au pire entre les variantes et 67,18 dB face à CUDA pour chacune. Le pic de différence de luma entre les variantes ZML vaut 3 codes à 10 bits et la plus forte erreur absolue moyenne sur une fenêtre de 16×16 vaut 0,0127 niveau à 8 bits. Face à CUDA, ces valeurs pour le centre sont de 50 codes et de 2,256 niveaux. Dans deux traces de 204 exécutions chaudes, le noyau de préparation des 28 canaux de D2 passe de 1,201 à 0,503 ms en médiane.

Sur **RIFE 4.25 lite FP16 en 1280×768**, le SDK CUDA calcule le warp du quatrième bloc sur les quatre pixels du centre de chaque cellule de 4×4. Le redimensionnement bilinéaire et l’arrondi FP16 du modèle sont conservés. Sur la RTX 4070 Ti, la médiane du réseau résident passe de 2,312 et 2,301 ms à 2,262 et 2,257 ms. En lecture composée ×8 sur l’animation, deux passages de neuf secondes par réglage en ordre ABBA font passer le pire taux d’images manquées de 6,02 à 4,59 % et le débit minimal de 91,93 à 92,23 images neuves/s. Les 185 images FFV1 sont identiques pixel pour pixel entre les deux graphes ZML et atteignent 59,63 dB au pire face à CUDA natif. Le module du lecteur normal reproduit les quatre noyaux du bloc et les 59 plans cuDNN de l’essai. L’option CMake `KIRARI_ZML_AUTO_RIFE425_LITE_D4=OFF` rétablit le graphe direct. [Sélection des arêtes du modèle](../src/gpu/zml/rife_d4_lite.cpp) · [Calcul ZML](../src/gpu/zml/sdk/motion_sampling.zig) · [Banc du réseau entier](../tools/bench/zml/d4_rife425_lite.py)

Pour 4.25, deux premiers blocs produisent un flot sur la même grille 96×160. Les additionner avant le redimensionnement commun réduit de 29 à 17 les opérations `gather` du HLO concerné. Dans l’essai sur les deux graphes du catalogue, le candidat matérialise aussi une carte FP32 de 1,875 Mio et le pire p95 passe de 15,94 à 16,91 ms. Le PSNR minimal entre les deux graphes ZML vaut 85,68 dB sur 47 images. Le modèle distribué garde les deux redimensionnements.

Sur 4.25, le profil des 54 inférences après amorçage donne en moyenne 7,46 ms de noyaux par inférence. Trois préparations des entrées prennent ensemble 3,17 ms. Un essai de réglage automatique du déroulage a donné exactement le même code machine et les mêmes dimensions de lancement pour ces trois noyaux. Dans les deux exécutables, 25 plans cuDNN sur 59 diffèrent et le second passage témoin a le meilleur p95. Le lecteur garde le déroulage d’origine. [Autotuning persistant XLA](https://openxla.org/xla/persisted_autotuning)

Le remplissage de 28 à 32 canaux à l'entrée de la convolution D1 de RIFE 4.25 semblait ramener le pire p95 de 15,82 à 15,57 ms, mais 30 autres convolutions avaient changé de plan cuDNN. Le banc `plans_cudnn.py` a reproduit les 59 plans du témoin, puis les 58 plans communs aux formes C28 et C32. En rejouant le même extrait dans l'ordre C28, C32, C32, C28, le pire p95 vaut 14,38 ms en C28 et 15,67 ms en C32. Les deux voies soumettent 314 images par passage et la pire des 47 images comparées entre elles atteint 95,92 dB. Le lecteur conserve les 28 canaux.

Sur le préfixe du même réseau, préparer D1 en bandes de 160 colonnes réduit le temporaire maximal de 86 555 392 à 76 389 636 octets. Les 27 525 120 mots FP16 de sortie sont identiques, y compris les deux colonnes adjacentes aux sept coutures. Les médianes de quatre séries de 100 inférences, dans l'ordre pleine carte, bandes, bandes, pleine carte, sont 3,107, 2,856, 2,826 et 2,675 ms. L'avantage en temps change de sens entre les deux paires.

Le calcul Nsight Compute de la convolution qui consomme les 28 canaux de RIFE 4.22 lite mesure 55,09 Mo lus, 23,32 Mo écrits et 76,46 % de succès en cache L2. Pour DistilDRBA, réunir les deux warps RVB et le masque de sortie dans un bloc ZML donne 8,28 ms au pire p95 du panoramique et 77,32 dB sur l’extrait d’animation. Cette réécriture reste expérimentale.

### ArtCNN en lecture réelle

Sur RTX 4070 Ti, ArtCNN C4F16 et C4F32 en mode FP16 atteignent la cadence du film de 119,88 i/s en 1080p → 4K. Le SDK ZML trace en FP16 les sept convolutions du corps et leurs cartes intermédiaires. Le calcul de luminance à partir du RVB, le redimensionnement de la chrominance et la recomposition couleur restent en FP32. Les fichiers ONNX gardent leurs poids FP32. Leur conversion dans le graphe arrondit les 12 340 paramètres du corps C4F16 et 47 715 des 47 716 paramètres C4F32. Le lecteur active ce choix pour l'empreinte SHA-256 exacte de chaque modèle à 1920 × 1080 sur cette carte. `-DKIRARI_ZML_ARTCNN_MIXED_BODY=OFF` rétablit le corps FP32. [Sélection et graphe](../src/gpu/zml/artcnn_body.cpp) · [Banc de précision](../tools/bench/zml/artcnn_precision.py)

Les quatre lectures par modèle alternent corps FP32, mixte, mixte, corps FP32 sur le même film de 8,992 s, avec composition Wayland, sans interpolation ni reprise de régions. Les valeurs sont les pires des deux passages de chaque bras en présentation automatique. Les PSNR sont ceux de la pire des 120 images FFV1 4K 10 bits, aux mêmes PTS que les références.

| ArtCNN | Images neuves/s FP32 → mixte | Rôle p95 FP32 → mixte | Sources manquées FP32 → mixte | PSNR mixte / TensorRT | PSNR mixte / ZML FP32 |
|---|---:|---:|---:|---:|---:|
| C4F16 | 95,06 → 119,76 | 10,87 → 4,20 ms | 11,96 → 0,28 % | 74,53 dB | 74,02 dB |
| C4F32 | 65,89 → 119,70 | 16,28 → 8,41 ms | 12,70 → 0,28 % | 73,36 dB | 73,74 dB |

Le pilote a bloqué la présentation d’un premier passage C4F16. La série automatique reprise en profils neufs figure dans le tableau. Une autre série C4F16, avec `--present=open` dans les deux bras, donne 94,28 → 119,37 images neuves/s et 10,90 → 4,33 ms au pire p95.

Sur des tenseurs résidents et une entrée capturée dans le lecteur, les médianes de C4F16 passent de 6,91–6,96 à 2,71 ms et celles de C4F32 de 12,75–13,21 à 4,75–4,82 ms. Les temporaires PJRT passent de 423 065 024 à 223 999 424 octets sur C4F16, et de 821 196 224 à 423 065 024 octets sur C4F32. Les deux programmes émis par le frontal de production ont le même SHA-256 que les modules StableHLO qualifiés. [Tests sur les modèles distribués](../tests/unit/zml_rt/artcnn_body_test.cpp)

La séance historique suivante compare le corps ZML FP32 à TensorRT sur le même film. Les deux moteurs utilisent des entrées et sorties FP16.

| ArtCNN | Images neuves/s ZML / TensorRT | Rôle p95 ZML / TensorRT | Sources manquées ZML / TensorRT | PSNR minimal |
|---|---:|---:|---:|---:|
| C4F16 | 94,17 / 114,56 | 10,95 / 9,69 ms | 10,84 / 4,45 % | 74,19 dB |
| C4F32 | 65,22 / 79,54 | 16,30 / 13,07 ms | 11,49 / 33,46 % | 74,67 dB |

Dans cette séance historique C4F32, le lecteur ZML soumettait 955 à 958 images en 14,63 s. TensorRT en soumettait 718 en 9,01 s.

Le banc `tete_artcnn.py` compose les deux premiers filtres linéaires en un 5×5 sur l’intérieur de l’image, avec le calcul d’origine sur les quatre bords. Il garde la première convolution pour le résidu. Malgré un minimum de 109,62 dB entre les deux exports ZML, C4F32 passe de 65,04 à 57,39 images/s et de 16,95 à 18,33 ms au p95. Le frontal livré garde les deux convolutions 3×3. [Architecture et poids ArtCNN](https://github.com/Artoriuz/ArtCNN/blob/main/README.md)

Sur une image réelle, une seconde expérience traite les sept convolutions 3×3 par bandes de 60 lignes avec sept lignes de halo de chaque côté. Le temporaire PJRT de C4F32 tombe de 821 196 224 à 56 400 388 octets. Les médianes de 40 passes montent de 12,70 et 13,04 ms à 13,41 et 14,27 ms. Les sorties aux coutures dépassent 112 dB contre le calcul dense. Ce découpage reste au banc.

### SPAN Balanced en lecture réelle

Sur les quatre SPAN FP16 du catalogue, le lecteur a ouvert séparément les voies SDK ZML et CUDA native en fenêtre SDR sur la RTX 4070 Ti. Chaque ligne retient le pire de deux passages préchauffés de 15 s par moteur, dans l'ordre ZML, CUDA, CUDA, ZML. La source à 59,94 i/s borne le débit natif. L'entrée mesure 1920 × 1080, la sortie 3840 × 2160. La réutilisation, les tuiles et l'interpolation sont désactivées. Le PSNR est celui de la pire des 60 images FFV1 10 bits aux mêmes horodatages par modèle.

| SPAN | ZML i/s | CUDA i/s | Manquées ZML / CUDA | p95 ZML / CUDA | PSNR minimal |
|---|---:|---:|---:|---:|---:|
| Balanced | 34,82 | 59,98 | 41,69 % / 0 % | 30,07 / 10,81 ms | 68,74 dB |
| Sharp1 Balanced | 34,73 | 59,99 | 41,87 % / 0 % | 30,36 / 10,53 ms | 69,25 dB |
| Performance | 57,58 | 59,98 | 3,79 % / 0 % | 18,22 / 6,12 ms | 70,41 dB |
| Sharp1 Performance | 56,91 | 59,99 | 4,90 % / 0 % | 18,44 / 6,43 ms | 70,09 dB |

Sur les deux Performance, le frontal SDK CUDA projette désormais les quatre entrées par bandes de 30 lignes en 1080p. Le banc alterné de 15 s par passage fige les 22 plans cuDNN communs dans les deux programmes. La projection est la seule convolution dont la forme change. Les temporaires passent de 447 903 488 à 298 605 056 octets. Les 180 images FFV1 10 bits de chaque modèle sont identiques entre pleine carte et bandes.

| SPAN Performance | Images neuves/s, pleine carte → bandes | Manquées | GPU p95 |
|---|---:|---:|---:|
| Performance | 57,01 → 59,79 | 4,68 → 0,33 % | 18,40 → 18,34 ms |
| Sharp1 Performance | 57,12 → 59,79 | 4,57 → 0,33 % | 18,44 → 18,05 ms |

`KIRARI_ZML_AUTO_SPAN_PERFORMANCE_PROJECTION_STRIPS=OFF` rétablit la projection pleine carte pour cette famille. Les deux temps GPU natifs de référence restent voisins de 6 ms.

Une source à 119,88 i/s lève la limite imposée par la source à 59,94 i/s. Les deux moteurs calculent SPAN pleine image sans réemploi. Sur deux lectures allant au bout des mêmes 8,992 s de contenu, Performance atteint **58,41 / 120,04 images neuves/s** pour ZML / CUDA natif et **18,09 / 5,41 ms** au p95. Balanced atteint **35,08 / 94,38 images/s** et **29,96 / 11,39 ms**. Performance ZML parcourt l’extrait en 16,40 s de temps hôte et Balanced en 27,02–27,14 s, contre environ 9 s pour CUDA natif. Le délai de lancement est compté séparément. La série d’exports sur 120 images avec ces deux mêmes exécutables donne 70,35 dB sur Performance et 68,66 dB sur Balanced au pire.

Le SDK réunit maintenant la projection 1×1 et la convolution de sortie 3×3 dans les mêmes bandes sur les deux Performance FP16 CUDA en 1080p. Il lit une ligne de halo de chaque côté et place les pixels dans leur ordre final dans la bande. Sur la totalité du film de 8,992 s à 119,88 i/s, le témoin projette déjà par bandes. Les lecteurs alternent deux fois par variante, en pleine image avec composition. Leurs 21 convolutions communes gardent les mêmes plans cuDNN. La projection et la convolution de sortie changent de forme.

| Modèle | Images neuves/s, projection seule → avec sortie | GPU p95 | Sources manquées | Durée hôte du candidat |
|---|---:|---:|---:|---:|
| Sharp1 Performance | 63,22 → 65,50 | 18,09 → 16,03 ms | 13,07 → 12,23 % | 14,38–14,44 s |
| Performance | 62,47 → 65,62 | 17,70 → 15,92 ms | 12,14 → 12,33 % | 14,40–14,44 s |

Les deux variantes rendent les mêmes 120 images FFV1 10 bits pour chacun des modèles. Les lectures avec les 21 plans figés et celles comparées à CUDA ci-dessus ont des caches et des exécutables distincts. `KIRARI_ZML_AUTO_SPAN_PERFORMANCE_PIXEL_STRIPS=OFF` garde uniquement la projection par bandes.

Le lecteur Release avec les réglages livrés et les plans choisis localement par XLA a ensuite exécuté les deux moteurs jusqu’à la fin du même film, sans variable de configuration CUDA ou XLA ajoutée. Les 8,992 s de contenu demandent 14,41 à 14,71 s à ZML et 8,98 s au natif. La première et la dernière présentation délimitent ces durées. Le PSNR porte sur les 120 images 4K de la première seconde exportée dans les deux voies.

| Modèle | Images neuves/s ZML / CUDA | Rôle GPU p95 ZML / CUDA | PSNR minimal |
|---|---:|---:|---:|
| Sharp1 Performance | 65,93 / 120,02 | 15,89 / 6,33 ms | 70,03 dB |
| Performance | 65,06 / 119,62 | 16,15 / 5,23 ms | 70,34 dB |

Un nouvel essai du même film à 119,88 i/s compare les deux Balanced à la version native avec le réglage de cache L1. Les journaux confirment l’emploi du chemin FP16 natif sur la RTX 4070 Ti. Chaque ligne retient le pire de deux lectures complètes et le PSNR minimal de 120 images 4K à 10 bits.

| Modèle | Images neuves/s ZML / CUDA | Rôle GPU p95 ZML / CUDA | PSNR minimal |
|---|---:|---:|---:|
| Balanced | 34,66 / 96,60 | 30,42 / 11,05 ms | 68,66 dB |
| Sharp1 Balanced | 34,61 / 97,05 | 30,48 / 10,97 ms | 69,17 dB |

La même fusion projection/sortie a été essayée sur Balanced C64, en réseau complet sur une image réelle. Les 33 plans cuDNN communs coïncident et les 24 883 200 valeurs FP16 de sortie sont identiques. Les médianes de 40 passes valent 24,66 et 24,80 ms pour la projection seule, 24,42 et 25,59 ms pour la projection avec sortie. Les temporaires baissent de 512 octets. Balanced C64 garde la projection seule par bandes.

Le frontal accepte aussi `--projection-gate-strips` pour différer la dernière porte du C48 jusqu’à la concaténation. Sur Performance et Sharp1 Performance, le HLO optimisé englobe six tranches, l’addition résiduelle, tanh, le produit et la concaténation dans une fusion. Les 23 plans cuDNN concordent entre témoins et candidats. Les 24 883 200 valeurs de chaque sortie sont identiques bit pour bit sur la sixième image capturée. Le temporaire passe de 298,61 Mo à 312,43–312,60 Mo. Sur Performance, les deux p95 du réseau seul passent de 13,79–13,88 à 14,38–14,40 ms. Sur Sharp1 Performance, les séries se recouvrent. La lecture garde la porte pleine image.

Sur Sharp1 Performance, une comparaison ZML contre ZML a isolé l'option XLA `--xla_gpu_experimental_enable_conv_fusion=true`. Elle fait passer la lecture de 56,97 à 50,75 i/s, les images manquées de 4,79 % à 15,14 % et le p95 de 18,39 à 20,83 ms. Le PSNR minimal entre les deux formulations vaut 74,31 dB sur 60 images. Le HLO des deux sources est exempt d’appel à un noyau Kirari. Dans les programmes compilés, les 23 appels cuDNN à la convolution classique et la fusion matricielle Triton du témoin laissent place à 23 appels au graphe cuDNN. Le réglage de lecture conserve la sélection standard de XLA.

Deux passages de 15 secondes en fenêtre SDR Wayland 1600 × 900 sur écran 165 Hz, RIFE et réutilisation SPAN désactivés. Le premier banc de transport utilisait une source à 59,94 i/s et retenait le pire passage pour chaque indicateur.

| Moteur et transport | Images neuves soumises/s | Images source manquées | SPAN p95 |
|---|---:|---:|---:|
| CUDA natif | 59,99 | 0,00 % | 11,05 ms |
| ZML, entrées GPU partagées et sortie GPU→GPU | 35,37 | 40,80 % | 29,32 ms |
| ZML, transport en mémoire hôte | 24,00 | 59,75 % | 35,87 ms |
| Vulkan | 27,40 | 54,07 % | 37,75 ms |

Sur 180 images FFV1 10 bits alignées, le PSNR minimal atteint 68,59 dB entre CUDA et le graphe ZML. Les deux voies de transport de ce graphe donnent les mêmes pixels. La concaténation de 253 Mio reste matérialisée avant la projection 1 × 1. Sa durée est d’environ 1,74 ms par image sur la trace. L’audit du graphe direct compte 35 appels cuDNN et 10 fusions XLA, dont deux Triton.

Pour situer la cible de calcul, le fil CUDA compte 4,365 ms de multiplications-accumulations pour le réseau Balanced avec ses 24 convolutions du corps en Winograd vertical F(2,3) et sommes FP16 à 160 TFLOP/s. À 90 % de ce débit, ce travail arithmétique représente 4,85 ms. Les transformées, les mouvements des tenseurs et l’affichage s’ajoutent au temps réel. Les médianes de réseau natif sur deux images se situent entre 9,11 et 9,62 ms. La lecture native à 118,72 i/s utilise un autre mode avec calcul par tuiles sur 58,2 % des images et source à 120 i/s. La comparaison pleine image ci-dessus désactive le réemploi des tuiles.

Un autre essai isolé de deux fois 10 secondes a remplacé les 24 convolutions du corps par des transformées Winograd F(2×2, 3×3) et des produits matriciels exprimés en StableHLO. Les 24 produits par lots figurent dans le module effectivement compilé. Sur 60 images alignées, le minimum est de 68,20 dB contre CUDA. Cette formulation atteint 7,39 images neuves/s et 144,08 ms au p95, contre 60,00 et 10,78 ms pour le natif dans la même séance. Elle reste désactivée. Le précédent essai Winograd vertical atteignait 64,75 ms au p95.

Le microbanc `tools/bench/zml/couche_span.py --bandes` isole la première 3×3 avec biais et ReLU du premier bloc, avec ses poids réels et une entrée FP16 résidente de forme 1×540×960×64. Sur deux séries de 100 inférences par variante, les médianes de la même séance atteignent 0,602 et 0,607 ms en direct, 1,692 et 1,704 ms en Winograd vertical pleine carte, et 1,155 et 1,158 ms sur 18 bandes de 30 lignes. Les temporaires PJRT occupent respectivement 256 octets, 253,125 Mio et 14,063 Mio. Les deux Winograd rendent exactement les mêmes 33 177 600 valeurs, y compris aux coutures. Leur écart quadratique moyen au sens RMS par rapport à la directe est de 0,000754. Les mesures de la couche excluent la composition.

Sur cette même couche, `tools/bench/zml/couche_span.py --carre4` transforme des tuiles de sortie 4×4 et exécute 36 produits en lot par bande de 20 lignes. Avec 15 appels de chauffe et 100 chronométrés par série, les médianes de la séance sont de 0,598 et 0,615 ms en direct, contre 1,871 et 1,872 ms en Winograd. Le temporaire compilé occupe 12 903 428 octets. Sur les 33 177 600 activations, l’erreur L2 relative est de 0,00326 et l’écart absolu maximal de 0,21875. La variante reste dans le microbanc. [Transformées de Lavin et Gray](https://openaccess.thecvf.com/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf)

Sur SPAN Performance C48, `tools/bench/zml/span_f4_vertical.py` reproduit le F(4,3) vertical du banc CUDA avec les coefficients 0,75 et 1,5. Sur la première convolution du premier bloc, le frontal Zig conserve les poids FP16 du modèle et découpe l'image en bandes de 36 lignes. Sur la même activation du film, la médiane de la couche passe de 0,476–0,478 ms en direct à 1,369–1,370 ms. XLA produit trois sites de convolution et neuf fusions, avec 26 543 364 octets temporaires contre 256 en direct. L'erreur L2 relative vaut 0,000316 sur les 24 883 200 activations, dont 1 382 400 aux coutures. Cette variante reste dans le banc de couche.

Sur le premier bloc Balanced complet, le 1×1 et les trois convolutions 3×3 peuvent aussi travailler sur six bandes de 90 lignes, avec trois lignes de halo. Les 33 177 600 sorties, dont 1 966 080 valeurs aux bords et aux coutures, sont identiques au témoin. Les médianes, 2,76 et 2,85 ms par bandes contre 2,68 et 2,88 ms en pleine carte, se recouvrent. Le temporaire compilé descend de 66 356 224 à 56 774 660 octets.

Le banc `couche_span.py --lignes` enchaîne les blocs 2 à 4 de Balanced par bandes de 45 lignes. Chaque 3×3 garde deux lignes en mémoire et chaque raccourci résiduel en garde trois. Le premier passage anticipe neuf lignes et le dernier restitue le bord inférieur. Sur la sortie résidente du bloc 1 issue d’une capture de lecture, les 33 177 600 mots FP16 produits sont identiques à la pleine carte, y compris aux onze coutures. Le temporaire PJRT passe de 132 713 472 à 43 509 252 octets. Les médianes de deux séries de 60 appels passent de 8,31–8,41 à 9,13–9,14 ms. Le corps de la boucle contient 12 convolutions cuDNN et 26 fusions XLA. Le lecteur garde les blocs pleine carte.

Le banc de la projection seule dans le lecteur (`KIRARI_ZML_SPAN_PROJECTION_STRIPS`) assemble les quatre sources en bandes de 30 lignes et garde la convolution 1×1 normale. Au pire de deux lectures de quinze secondes, la première mesure donne 34,89 images neuves/s, 41,58 % de sources manquées et 30,28 ms au p95. Le PSNR minimal est de 64,43 dB sur les 180 images 4K. Un second banc en ordre alterné donne 34,93 images neuves/s en pleine carte et 34,92 par bandes, avec des p95 de 30,12 et 29,94 ms. Dans le HLO optimisé, la concaténation passe de `1×540×960×256` à `1×30×960×256` et le temporaire de l’exécutable de 632,82 à 379,70 Mio. Le lecteur active les bandes sur le SDK CUDA pour Balanced b8f64 FP16 en entrée 1920×1080. `KIRARI_ZML_AUTO_SPAN_BALANCED_PROJECTION_STRIPS=OFF` redonne le témoin pleine carte.

Les empreintes des 180 images décodées, avec leurs horodatages, sont identiques entre les deux exports ZML pleine carte et par bandes. Sur un autre microbanc du premier bloc, exprimer le produit matriciel de la porte, tanh, le résidu et la multiplication finale sur une même grille 2D donne 2,654 ms par bloc contre 2,663–2,672 ms en direct. Le HLO montre que le produit matriciel absorbe le biais et tanh. La multiplication par la somme des deux autres entrées reste dans une fusion séparée. La passe GEMM des sources publiques référencées par le SDK accepte comme entrée latérale de cet épilogue un paramètre ou certaines diffusions. Une addition calculée dans le graphe ne satisfait pas ce critère. [Fusion GEMM dans OpenXLA](https://github.com/openxla/xla/blob/fc31477/xla/backends/gpu/transforms/gemm_fusion.cc)

Sur le vrai premier bloc de Balanced C64, le code machine de la porte contient 32 instructions statiques `HMMA.16816.F32` avec la configuration ordinaire du produit matriciel et 32 `HMMA.16816.F16` en demandant l’accumulation FP16 à Triton. Les trois convolutions voisines conservent leurs plans cuDNN. Les médianes de deux séries vont de 2,46 à 2,70 ms avec sommes FP32 et de 2,60 à 2,65 ms avec sommes FP16. Le microbanc écarte la passe `algorithm-checker` dans les deux bras, car la liste des algorithmes acceptés par cette version de XLA omet cette combinaison. Le lecteur conserve les vérifications ordinaires. [Contrôle des algorithmes dans OpenXLA](https://github.com/openxla/xla/blob/fc31477d352ed8edc223648524382f9b10a7699b/xla/service/algorithm_util.cc)

La convolution 3×3 C1 a également été exprimée avec neuf phases spatiales dans le graphe. Le binaire chargé contient alors des instructions `HMMA.16816.F16`, contre `HMMA.16816.F32` pour la convolution dense inspectée sous Nsight Compute. Les 54 tranches imposent une écriture intermédiaire de 569,53 Mio cumulés avant les produits matriciels. La médiane de C1 passe de 0,61–0,62 ms à 2,08–2,10 ms sur 100 inférences par série. L’écart L2 relatif atteint 0,000837 sur les 33 177 600 activations de sortie. Le lecteur conserve la convolution dense.

### SPAN Balanced sur une zone active

Le lecteur CUDA a enregistré les six entrées RVB FP16 après prétraitement, leurs dates, les décisions de réemploi et le masque de tuiles. L'image 1 est calculée entièrement puis réutilisée sur les images 2 à 5. À 42 ms, l'image 6 comporte 95 tuiles actives sur 2040. Le banc régional conserve la sortie de ZML pour l'image 1 sur le GPU et parcourt six fenêtres de 200 × 360 pixels pour l'image 6, halo compris. Le greffon compile 35 appels cuDNN dans chacun des deux programmes à partir des mêmes poids. Le StableHLO émis par le frontal contient zéro appel à un noyau personnalisé.

Sur cette image, la médiane passe de 24,73–24,79 à 5,17–5,18 ms. Le pire p95 passe de 25,36 à 6,50 ms. Les 23 758 848 mots FP16 hors cœur sont identiques à leur valeur précédente. Face au calcul dense de l'image 6, le PSNR vaut 57,79 dB sur l'image entière, 87,18 dB dans le cœur et 86,88 dB aux coutures. Le pire bloc de 16 × 16 vaut 0,918 niveau de luma avec la pondération BT.709 et la plage limitée du lecteur, sous la garde d’un niveau. La moyenne RVB multipliée par 255 du premier contrôle est une autre métrique. La part active de ce masque est de 4,66 %. Les durées ci-dessus couvrent une exécution résidente de ce cas.

En lecture, le contrôle de chaque décision régionale sur 14 secondes à 24 i/s trouve 26 images sur 47 avec au moins un bloc au-dessus du seuil d'un niveau de luma. Le maximum atteint 1,430 niveau sur un bloc repris de l'image précédente. Les maxima sur les bords recalculés et à l'intérieur des cœurs sont de 0,031 et 0,043. Ramener le seuil d'entrée de 0,70 à 0,25 écarte les 337 décisions de la première séquence et les 438 de la seconde. La lecture garde le calcul dense par défaut.

### DistilDRBA et GMFSS

Sur DistilDRBA, la projection RVB alimente trois branches du réseau. La conserver une seule fois dans le graphe Zig ramène les temporaires de 43 898 112 à 37 999 872 octets. Les médianes de deux séries sur le réseau complet passent de 0,986 et 0,991 à 0,946 et 0,947 ms. Sur la paire contrôlée à l’instant du microbanc, la sortie FP16 est identique bit pour bit. Sur le panoramique en lecture composée de dix secondes par passage, le pire p95 passe de 7,84 à 7,26 ms, avec 240 à 242 mesures du rôle, 47,70 à 47,71 images neuves/s et 0,62 % de sources manquées dans les deux cas. Sur 47 images FFV1, le pire PSNR entre les deux variantes ZML est de 77,19 dB sur le panoramique. Sur l’extrait d’animation, le minimum contre le témoin natif atteint 77,33 dB. Le lecteur active la projection partagée pour le DistilDRBA FP16 reconnu à deux entrées de sept plans en 1280×768 sur le SDK CUDA. `KIRARI_ZML_DRBA_SHARED_PROJECTION=OFF` reproduit le témoin.

Une comparaison directe sur le même panoramique utilise le lecteur CUDA fusionné figé lors de la série d’animation. Sur les quatre lectures alternées, le rôle interpolation compte 241 mesures ZML et 242 CUDA par passage, sans coupe ni dessin tenu. Au pire, ZML soumet 47,70 images/s avec 0,62 % de sources manquées et un p95 de 7,41 ms. Le natif soumet 48,01 images/s avec 0,00 % manquées et un p95 de 4,45 ms. Les médianes du rôle valent 4,18–4,93 ms sur ZML et 1,46–1,66 ms sur le natif. La pire des 47 images FFV1 comparées sur ce panoramique atteint 69,56 dB.

Sur un extrait d'animation à 23,976 i/s, en 1080p avec interpolation ×8 et sortie Wayland à 165 Hz, ZML et CUDA soumettent chacun au moins 162,95 images neuves/s sur deux passages de six secondes. Les retours de présentation datent entre 856 et 858 images neuves sur 979 à 981 soumises par passage. Sur la partie datée, les deux moteurs atteignent 165,00 images neuves/s. Le p95 du rôle interpolation atteint 5,73 ms pour ZML et 1,02 ms pour CUDA, avec respectivement 0,20 % et 0,10 % de sources manquées. Les 185 images de l'export FFV1 10 bits aux mêmes dates donnent un PSNR minimal de 74,69 dB. La restauration des lignes et l'agrandissement sont désactivés dans les deux lecteurs.

Le lecteur réutilise maintenant les commandes Vulkan de transfert de propriété lorsque la liste des tampons et la direction sont identiques. Chaque soumission fournit ses propres valeurs de sémaphore. Pour isoler ce changement, `modeles.py --meme-programme-pjrt` vérifie les empreintes du greffon, du StableHLO et du programme compilé puis fait tourner les deux lecteurs dans l'ordre témoin, candidat, candidat, témoin. Sur DistilDRBA, le pire p95 passe de 8,38 à 5,58 ms (médianes de 4,32 et 4,40 ms pour le témoin, 3,81 et 3,15 ms pour le candidat). Sur RIFE 4.25, le pire p95 passe de 13,17 à 10,09 ms. Le débit reste proche de 47,8 images/s et chacun des deux modèles donne 47 images exportées identiques entre lecteurs.

Une capture Nsight Systems séparée de DistilDRBA enregistre 147 sorties GPU dans chaque lecteur. Entre la troisième et l'avant-dernière copie de sortie, soit 143 intervalles, le nombre d'appels Vulkan du processus à `vkCreateCommandPool` passe de 296 à 11, `vkAllocateCommandBuffers` de 297 à 11, `vkCreateSemaphore` de 296 à 10 et `vkQueueSubmit` de 859 à 716. Ce comptage inclut les autres opérations Vulkan du lecteur. Les chronométrages du paragraphe précédent proviennent des lectures sans instrumentation. [Trace Vulkan dans Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#vulkan-trace)

Sur DistilDRBA ×8 en 1080p, une capture Nsight de 980 inférences recharge le même exécutable PJRT que la mesure de référence. L’intervalle entre le premier et le dernier noyau vaut 3,114 ms en médiane et 3,689 ms au p95. Les noyaux occupent respectivement 3,013 et 3,578 ms. La dernière fusion calcule les deux warps et la sortie RVB en 1,073 et 1,622 ms. La copie de sortie GPU→GPU prend 0,029 et 0,031 ms.

Une formulation groupant les trois couleurs par pixel partage les coordonnées entre les canaux. Le réseau complet descend de 3,038 ms pour le dernier témoin chaud à 2,625 ms en médiane, avec les mêmes 37 plans cuDNN et les 6 266 880 mots FP16 identiques sur l’entrée mesurée. En lecture composée, les deux p95 du témoin sont de 5,522 et 5,573 ms, et ceux du candidat de 5,454 et 5,720 ms. Les trois lecteurs rendent 185 images aux mêmes dates, identiques entre témoins ZML et à 74,69 dB au pire contre CUDA. L’option `KIRARI_ZML_DRBA_OUTPUT_PIXEL` reste désactivée.

Le banc `encodage_distil.py` conserve le lot des trois encodages dans le client PJRT entre les appels. L'encodeur vérifié ne lit que les trois tranches RVB. Sur le même trio à deux instants FP16 (0,375 et 0,5625), les 2 949 120 mots par sortie sont identiques. Les 37 plans cuDNN se retrouvent dans les deux programmes (4 pour l'encodeur et 33 pour le reste). Les dernières médianes de référence sont de 0,995 et 0,997 ms, celles avec encodage déjà disponible de 0,953 à 0,985 ms. L'encodage isolé prend 0,370 à 0,524 ms en médiane. Cette expérience mesure le calcul entre appels et son coût de préparation séparément. Elle ne modifie pas le traitement des changements de trio du lecteur.

Sur ces deux mêmes instants, le graphe du banc `cote_distil.py` choisit les entrées RVB et les traits d’après le coefficient temporel du réseau. Les 37 plans cuDNN et les 2 949 120 mots FP16 de sortie restent identiques. Les temporaires montent de 37 999 872 à 43 898 368 octets. À u = 0,375, les deux médianes de la variante valent 1,105 et 1,083 ms, contre 0,998 ms pour le dernier témoin chaud. Le lecteur conserve le coefficient temporel du modèle dans le graphe partagé.

Un second essai déplace la conversion FP16→FP32 du warp central après la sélection des quatre voisins. Le HLO compilé remplace la carte RVB FP32 de 11 796 480 octets par une carte FP16. Le total temporaire passe de 37 999 872 à 32 101 376 octets. Sur la même paire du panoramique, les 2 949 120 mots de sortie sont identiques et les médianes de deux séries de 100 inférences descendent de 1,010–1,013 à 0,982–0,984 ms. Le frontal dispose de l’option de construction `KIRARI_ZML_DRBA_GATHER_FP16` pour mesurer sa lecture composée sur CUDA.

Avec les mêmes 37 plans cuDNN, les deux lecteurs donnent 47 images identiques. Le pire p95 passe de 6,66 ms avec le réglage de lecture à 8,26 ms en activant cette option. Le débit reste compris entre 47,70 et 47,72 images/s, à 0,62 % de sources manquées. Les comparaisons indépendantes au témoin CUDA natif donnent 69,56 et 69,52 dB respectivement pour la pire image. L'option reste désactivée.

Sur le dernier bloc de DistilDRBA, le microbanc Winograd vertical extrait l’activation d’entrée du panoramique et vérifie les huit convolutions résiduelles contre le fragment ONNX. Les médianes de la séance passent de 0,171–0,174 ms à 0,166–0,169 ms. Le temporaire PJRT augmente de 739 328 à 5 898 240 octets et l’erreur L2 relative par rapport à l’ONNX atteint 0,00126. La voie de lecture conserve le bloc direct.

Sur le microbanc d’appariement GMFlow 36×60, normaliser et projeter séparément les deux sens avant la concaténation donne 0,756 ms dans les deux séries contre 0,790 et 0,815 ms pour le graphe original. Les deux exécutables calculent une seule corrélation 2160×2160 et réservent tous deux 74 649 616 octets temporaires. Le flot aller est identique et l’écart maximal du retour vaut 0,00281 pixel. Le lecteur garde le graphe GMFSS qualifié en lecture.

Sur le fragment GridNet 04 de GMFSS Union, la reformulation ZML mesurée sur les entrées capturées en 1920×1152 donne 4,65 ms en médiane sur les deux passages, contre 4,79 et 4,84 ms. Le temporaire compilé augmente de 393 728 octets et l’écart absolu maximal de la sortie vaut 0,000244. La lecture GMFSS utilise encore le graphe qualifié d’origine.

Sur le matcher local de GMFlow, `appariement_local.py` rejoue une fenêtre de 9×9 sur les entrées capturées à 960,40 s, en 120×72 dans les deux sens. La formulation directe du SDK prend 0,561 et 0,585 ms en médiane sur deux séries de 80 inférences. Regrouper 2×8 requêtes en produits matriciels porte le calcul de 179 à 354 millions de multiplications-accumulations, le temporaire de 11,14 à 176,95 Mo et les médianes à 0,992 et 0,998 ms. Les deux sorties ont un écart RMS compris entre 0,0082 et 0,0086 pixel par sens face à l’ONNX. Ces mesures concernent le fragment d’appariement.

La paire GMFlow d’Union s05 contient 179 produits matriciels. Ses conversions explicites en FP16, sa corrélation FP32 et le repère des coordonnées sont sensibles à l’ordre des opérations. Dans le SDK Zig, les poids bilinéaires du redimensionnement `align_corners` utilisent la division FP32 aux coordonnées ONNX. Pour les warps à remplissage nul, la marge s’ajoute à l’indice entier après le calcul du poids. Avec les mêmes plans de convolution, ces deux corrections seules rendent les 14 tenseurs de chaque paire identiques au frontal StableHLO de référence, sur le panoramique 640×360 et sur trois captures 1080p, soit 381 335 040 mots FP16 pour ces trois dernières.

Le SDK place ensuite une barrière d’optimisation sur les arrondis en FP16 explicitement demandés par l’ONNX. Avec les plans de convolution figés et la même fusion, les 95 images FFV1 donnent au pire 60,69 dB face au graphe de référence. Le banc `tools/bench/zml/rendu_gmflow.py` compare le rendu des deux paires au même ONNX Runtime CPU, avec un splat et une fusion communs. Sur l’entrée reconstruite du panoramique, le PSNR passe de 49,87 à 50,16 dB. Sur les trois captures 1080p prélevées dans le lecteur, il passe respectivement de 54,92 à 55,42, de 68,40 à 68,40 et de 75,06 à 75,06 dB.

Deux compilations à froid du seul frontal historique, à source identique, ont donné des exports distants de 37,42 dB au pire. Une comparaison en profils neufs du SDK corrigé et du frontal historique donne 42,58 dB sur les 95 images du panoramique. Les plans cuDNN choisis à la compilation et les additions du splat à l’exécution peuvent chacun contribuer à l’écart. [Autotuning et reproductibilité dans OpenXLA](https://openxla.org/xla/determinism)


Sur trois paires du panoramique, une sonde a conservé les entrées juste après leur préparation sur GPU. Les trois PTS correspondent à des pixels d’entrée identiques avec l’extension de bord du lecteur. Sur cette entrée, le SDK en précision ordinaire donne 35,45 dB contre l’ONNX CPU, avec un splat et une fusion communs. Le code machine des 107 fusions de produits matriciels inspectées contient des instructions TF32. Le frontal demande désormais la précision `HIGHEST` aux 36 convolutions et aux 180 contractions à deux opérandes FP32 du graphe de paire. En autotuning normal, le rendu remonte à 45,25 dB. Les 132 symboles GEMM désassemblés emploient des instructions FFMA. Les appels aux bibliothèques cuDNN et cuBLAS sont comptés séparément. Sur la capture 1080p à 960,40 s, le même réglage donne 56,71 dB contre 55,42 dB avec les arrondis seuls. Le seuil de 50 dB face à l’ONNX reste ouvert sur le panoramique. [Précision des contractions StableHLO](https://github.com/openxla/stablehlo/issues/755)

Avec cette précision, deux compilations indépendantes du lecteur donnent 69,74 dB au pire entre leurs 95 images FFV1, des débits de 47,21 à 47,56 images neuves/s et des p95 d’interpolation de 11,84 à 14,00 ms sur les quatre lectures. Cette comparaison mesure la répétabilité entre programmes. Le test face à l’ONNX mesure la fidélité des pixels.


Sur le panoramique GMFSS à 640×384, deux configurations de l'oracle ONNX Runtime 1.31.1 CPU donnent elles-mêmes 46,16 dB entre le réglage stock et les passes de graphe désactivées. Réduire les threads CPU de quatre à un donne 108,21 dB. Les huit sorties image/caractéristiques sont identiques entre niveaux de passes. Les six cartes de mouvement diffèrent, jusqu'à 2,89 pixels sur la composante verticale du flot retour à demi-échelle. Le graphe SDK HIGHEST atteint 45,25 dB face à l'oracle stock et 44,55 dB face aux passes désactivées, avec le même splat, la même fusion CPU et le même recadrage. Le seuil de 50 dB reste ouvert sur cette paire. Le banc `oracle_ort_gmflow.py` enregistre séparément les comparaisons de chaque composante.
