# DistilDRBA en noyaux natifs

Sur Ada en 1920×1088, la chaîne fusionnée peut utiliser le dernier bloc CUDA natif avec Winograd
vertical. Dans la voie mixte, le réglage « Dernier bloc DistilDRBA natif (essai) » est coché par
défaut. L'encodeur et les deux premiers blocs restent sous TensorRT. Le chargement y garde aussi le
dernier bloc lorsque les contrôles de rendu, de réemploi ou de durée échouent.

## Voie CUDA directe

Le réglage facultatif « Interpolation CUDA directe (essai) » charge directement les poids officiels
Équilibre du modèle en 1920 × 1088, avec les trois blocs et l'encodeur en CUDA. En lecture 1080p,
cette forme s'obtient avec l'estimation du mouvement en pleine taille. Le backend vérifie les
empreintes des quatre morceaux ONNX et la géométrie. Il ne construit ni ne compare de plan TensorRT
pour DistilDRBA. Le module NVIDIA garde sa dépendance au runtime TensorRT pour les voies mixtes.
Un refus de forme remonte au lecteur, qui peut poursuivre à la cadence source. Pour retrouver la
voie mixte sur d'autres tailles, décocher l'essai.

Sur deux trios réels et deux instants, le CUDA direct donne 78,85–81,56 dB RGB face à ONNX Runtime
CPU, avec un écart maximum de 0,005371. L'écart maximal de flot vaut 0,125 pixel. Le réemploi du
bloc B0 et le nouveau pavage des résidus B0/B1 conservent les 66 846 720 codes FP16 comparés.
Le chargement isolé après création du backend et avec les modèles dérivés en cache passe de 625
à 110 ms sur le premier prototype. Le banc du lecteur a aussi confirmé l'absence de plans Distil
dans un cache d'artefacts vierge avec l'essai activé.

| Graphe GPU à 1920 × 1088 | Mixte | CUDA direct | Borne Winograd + fusion | Part mixte → directe |
| --- | ---: | ---: | ---: | ---: |
| À froid, avec l'encodage des trois images | 0,704 ms | 0,772 ms | 0,197 ms | 28,00 % → 25,55 % |
| Avec réemploi, nouvel instant | 0,488 ms | 0,600 ms | 0,166 ms | 33,94 % → 27,64 % |

Le banc A/B/B/A mesure du premier noyau de préparation à la sortie flot/masque sur la RTX 4070 Ti.
La borne conserve les sommes FP32 et suppose le Winograd F(2×2,3×3) et les fusions par bloc. En
lecture ×8 avec SPAN Performance vers la 4K et composition Wayland, le mixte présente 164,788–165,000
nouvelles images/s et le direct 164,999, sans abandon sur les quatre passages. Le mixte conserve
l'avantage en temps GPU par interpolation.

## Bancs des blocs et de l'encodeur

Pour le bloc 0, un banc de convolution directe charge le halo de ses 192
canaux une fois par bloc CUDA. Sur quatre entrées réelles, les huit états
résiduels et la projection restent identiques à la référence directe à
sommes FP32. Le meilleur réglage fait passer les huit résidus de 181,38 à
134,88 µs sur RTX 4070 Ti. Le bloc CUDA mesuré de bout en bout reste autour
de 190 µs, contre 81,87 µs pour le bloc TensorRT. La voie directe reste au
banc en attendant une réduction du coût de ce bloc.

Le banc Winograd F(2 × 2, 3 × 3) des mêmes résidus passe de 157,84 à
74,97 µs, à sommes FP32 sur des transformées rangées en FP16. Le bloc
entier atteint 119,26 µs dans cet essai, contre 81,59 µs pour TensorRT.
Sur les quatre captures, le bloc Winograd donne 63,31–65,53 dB face à la
référence ONNX CPU. La compensation des poids transformés donne
64,01–66,04 dB et porte les résidus à 124,39 µs. Le rendu de la chaîne
reste à vérifier avant toute activation.

Découper K en six tranches garde les opérandes FP16 et les sommes FP32,
avec un ordre d'addition différent. Les huit résidus passent de 151,77 à
87,67 µs lors de passages stables, mais le bloc entier reste à 135,03 µs
contre 85,65 µs pour TensorRT. Cette variante reste au banc.

La fusion du rangement de l'encodeur et de sa sortie après warp a donné
24,05 à 13,01 µs au banc isolé, avec 24 675 840 valeurs identiques sur trois
résolutions. La lecture appariée à ×2 avec SPAN Performance donne environ
48 nouvelles images/s sous Wayland aux deux réglages. Les mesures GPU du
moteur complet se recouvrent. Cette fusion reste au banc.
Les mesures et le détail des contrôles sont dans [la fiche de la chaîne fusionnée](drba-fusionne.md).

Un prototype de l'encodeur réutilise les convolutions RIFE à sommes FP32 puis range les
16 canaux en plans pour la colle DistilDRBA. Sur deux trios réels de 480 × 272, les
6 266 880 valeurs d'activation par trio donnent 86,25 à 89,71 dB face à ONNX Runtime
CPU FP32. Le pire écart absolu est de 0,001664. Les passages stabilisés de l'encodeur seul
mesurent 101,95 à 102,30 µs de médiane, rangement compris. Des passages ont aussi
mesuré 0,46 à 0,71 ms. Il faut vérifier le coût en lecture et l'effet des écarts
d'encodage sur le résultat de la chaîne entière.

Le même banc groupe maintenant les trois images dans les quatre lancements de
l'encodeur. Les 12 533 760 valeurs des deux trios sont identiques bit à bit à
celles des trois appels séparés, avec ou sans écriture planaire dans la dernière
convolution. Sur les passages chronométrés en alternance, la médiane baisse
d'environ 97 à 77 µs, rangement final compris. Le changement concerne l'encodage
à froid. Le rejeu à temps seul réutilise déjà les encodages.

### Colle B0→B1 en trois groupes indépendants

À 1920 × 1088 sur Ada, la transition planaire B0→B1 répartit les plans de
contrôle et les deux warps en trois groupes. Les groupes relisent les mêmes
valeurs et conservent les arrondis du Resize, du flot et des warps. La
variante est basée sur la disposition étudiée dans le banc Triton. La
comparaison dans le lecteur, avec les voies image et mouvement, couvre
133 693 440 codes FP16 identiques, avec et sans réemploi et pour les deux
rangements temporels testés. Le chemin compact et le rangement X2 HWC8
gardent leur implémentation dédiée.

Avec SPAN Performance et une sortie Wayland à 165 Hz, la médiane des nœuds
B0→B1 des quatre passages alternés passe de 19,200 à 13,248 µs. En prenant
une seule visite des secteurs utiles des quatre captures et 2,748 To/s au
L2, leur borne de trafic vaut 2,191 µs, soit **11,4 % → 16,5 %** sur cette
étape. Le modèle exclut les instructions de calcul et le reste du réseau.
Les quatre passages affichent environ 165 nouvelles images/s, sans abandon.
Le binaire final confirme 13,248 µs sur 671 nœuds, avec 164,84 nouvelles
images/s et un abandon sur cette passe.

### Entrée B2 à 510 tuiles

La colle XOR compacte de B2 avec convolution de pas 2 utilise une tuile
de 16 × 4 sur Ada 60 SM en 1920 × 1088. Les 510 tuiles couvrent exactement
les 240 × 136 sorties et réutilisent la disposition des fragments fixes
du cache de la tuile large. Les autres géométries et l'entrée sans colle
compacte reprennent leur chemin habituel. Les essais à froid et avec
réemploi, en mode image et mouvement, comparent 133 693 440 codes FP16
bit à bit dans le lecteur.

Sous SPAN Performance ×8, les quatre passages appariés donnent une
médiane B2 de 93,944 → 84,159 µs. Pour comparer au même palier, la
géométrie compacte représente 3,159 GFLOP à froid et 1,805 GFLOP à
chaud avec ses sommes FP32. Au débit Tensor de 86,75 TFLOP/s, ses bornes
sont de 36,41 et 20,81 µs. La part atteinte sur ce noyau passe de
**33,5–33,9 % à 36,9–37,1 %** à froid et de **22,7–22,9 % à 25,4 %**
à chaud. La borne décrit les produits Tensor de l'entrée B2 fusionnée.
Le Winograd des 3×3 de pas 1 concerne les couches suivantes du bloc.
Le p95 du moteur entier passe de 0,73 à 0,71–0,72 ms. Les deux passages
candidats affichent 165,00 nouvelles images/s sans abandon via Wayland.

### Projection B2 par paires de phases

Dans le bloc final natif 1920 × 1088 sur Ada 60 SM, chaque CTA calcule
deux phases horizontales de la transposée et partage le fragment
d'activation central. Le noyau conserve l'ordre K16 puis prise de chaque
phase et le pas de sortie à 24 demi-flottants. Les vingt plans de
mouvement utiles et les quatre zéros de remplissage restent identiques.
La qualification compare 267 386 880 codes FP16 du lecteur sur deux
scènes, en mode image et mouvement, avec deux formes de tuiles candidates.

Avec SPAN Performance ×8 et la composition Wayland, la transposée passe
de 20,592 à 14,928 µs en médiane. Ses 0,501 GFLOP utiles à sommes FP32
donnent une borne Tensor de 5,779 µs à 86,75 TFLOP/s, soit **28,07 %
→ 38,71 %** à borne identique pour la projection seule. Les deux
passages retenus affichent 165,00 nouvelles images/s sans abandon.
Le binaire final confirme 14,912 µs de médiane sur 670 nœuds et
164,999 nouvelles images/s, sans abandon.

## Proposition initiale du 19 novembre 2026

Le plan ci-dessous visait les trois blocs et l'encodeur. Les noms, estimations et options correspondent
à ce projet initial.

### Gain attendu à l'époque

Chaîne DistilDRBA fusionnée (banc 4070 Ti, 1088×1920) : 0,92 ms par image créée, dont 0,59 ms pour les trois
blocs TensorRT (0,085 / 0,125 / 0,379), 0,11 ms d'encodeur (sauté en temps seul, D6) et 0,15 ms de colle.
À 20 % près sur les convolutions : 0,1 à 0,15 ms par image créée. Le vrai gain : ne plus construire les quatre
moteurs (38 s au premier lancement), une fois le natif tenu.

### Structure (routineLife1/DistilDRBA, distilDRBA_v2_lite.py)

- Head : cnn0 3→16 pas 2, cnn1 et cnn2 16→16, cnn3 ConvTranspose 16→16 (4×4, pas 2) ; LeakyReLU 0,2 sur les
  trois premières. Même forme que RIFE, mais 16 sorties au lieu de 4.
- IFBlock comme RIFE : conv0 (c/2 pas 2, puis c pas 2), 8 ResConv à bêta, lastconv ConvTranspose c→52.
  c = 192, 128, 96 ; entrées 58 (X_0) et 109 (X_1, X_2, flot compris).
- `prepare::rife_native_layers` lit déjà cette chaîne ; `rife_native_check(layers, 3, 16)` pour l'encodeur,
  `(58 ou 109, 52)` pour les blocs.

### Intégration envisagée

- Formats en plans (kernels/spec/drba_fused.h) : X_k en NCHW (58 ou 109 plans), Y_k en NCHW (52 plans à la
  moitié de X_k), l'encodeur lit trois plans dans X_2 (plans 3i à 3i+2) et écrit 16 plans (9 + 16i) au même
  format. Dans `rife_native_kernels.cuh` :
  - `In::kPlanes` : la tranche [c0, c0+Kc) lue plan par plan (un demi-flottant par fil, fils voisins en x),
    zéros au-delà des canaux réels (paramètre in_ch) ;
  - `Out::kPhasePlanes` : la phase d'une transposée écrite plan par plan ; `reduce_kernel` aussi.
- `RifeNativeNet::create_planar(pieces, encoder h×w, blocs (canaux, h, w), acc32, error)` : encodeur kImage →
  kSame → kSame → kPhasePlanes (16 canaux) ; bloc kHalfPlanes → kHalf → 8 kSame résiduels → kPhasePlanes (52).
  out_plane = 4·oh·ow dans les deux cas.
- `DrbaFusedEngine` : native_, try_native, trial_output comme RifeFusedEngine (`rife_fused_native.cpp`) ; trois
  appels d'encode (un par image), la zone de travail au bas de l'arène (`arena->used`).
- `--drba-native`, bit d'essai suivant libre (RIFE 1<<6, GMFSS 1<<7), kBackendAbi + 1, kFlags et kTrialOptions
  recomptés, options.cpp ≤ 800 lignes.
