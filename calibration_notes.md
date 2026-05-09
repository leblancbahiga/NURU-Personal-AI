# Notes de Calibration — ComplexityClassifier de NURU

Date : 05 Mai 2026
Analyse basée sur 22 tests représentatifs + historique 65+ entrées (complexity_log.json)

---

## 1. Analyse de l'Historique (complexity_log.json)

65 entrées analysées, couvrant des interactions du 3 au 5 Mai 2026.

### Constats clés :

- 45+ requêtes triviaires ("Bonjour", "hello", "test", "allo", "1") routées vers **N3 (Cloud)** alors qu'elles devraient être N1 (RAG local). Latence élevée (2-17s) pour des réponses simples.
- Plusieurs requêtes nécessitant Internet ("Cherche-moi les dernières nouvelles sur l'intelligence artificielle", "Quelle est la situation actuelle en Iran ?") routées vers **N1 (RAG local)** au lieu de N3.
- Scores de complexité historiques non enregistrés dans le log — seul le niveau choisi est présent.
- Les routages sont incohérents : une même requête ("Qui suis-je?") passe alternativement par N1, N2, N3.

### Distribution des niveaux historiques :
| Niveau | Occurrences |
|--------|-------------|
| N1     | ~28         |
| N2     | ~8          |
| N3     | ~29         |

---

## 2. Résultats des Tests (22 scénarios)

### Configuration Actuelle
**Poids** : w_rag=0.25, w_keywords=0.40, w_length=0.20, w_structure=0.15  
**Seuils** : N1<0.36, N3>0.42

| Requête | Attendu | Obtenu | Score | Breakdown |
|---------|---------|--------|-------|-----------|
| Bonjour | N1 | N1 | 0.355 | rag=0.8 kw=0.5 len=0.0 struct=-0.3 |
| hello | N1 | N1 | 0.355 | rag=0.8 kw=0.5 len=0.0 struct=-0.3 |
| Qui es-tu ? | N1 | **N2** | 0.363 | rag=0.8 kw=0.333 len=0.0 struct=0.2 |
| Explique la photosynthèse en détail | N2 | **N1** | 0.355 | rag=0.8 kw=0.5 len=0.0 struct=-0.3 |
| Écris un script Python pour parser du JSON | N2 | **N1** | 0.355 | rag=0.8 kw=0.5 len=0.0 struct=-0.3 |
| Compare le riz irrigué et le riz pluvial (RAG=0.45) | N3 | **N2** | 0.397 | rag=0.7 kw=0.667 len=0.0 struct=-0.3 |
| Définition simple RAG bas (0.20) | N2 | **N1** | 0.338 | rag=0.7 kw=0.333 len=0.0 struct=0.2 |
| Question RAG score moyen (0.55) | N2 | **N1** | 0.328 | rag=0.45 kw=0.5 len=0.0 struct=0.2 |

**7 erreurs de routage sur 22 tests (32% de mismatch).**

---

## 3. Analyse des Problèmes Identifiés

### Problème 1 : Score de structure négatif (BUG)
Dans `score_ponctuation_structure()`, la formule `(n_questions - 1) * 0.3` pénalise les requêtes SANS point d'interrogation.
- Queries sans `?`: structure = -0.3
- Queries avec 1 `?`: structure = 0.0
- Queries avec 2 `?`: structure = 0.3

**Impact** : Toute requête sans question ni ponctuation perd 0.3 en score, ce qui la fait artificiellement descendre vers N1. Exemple : "Explique la photosynthèse en détail" → score 0.355 au lieu de ~0.385.

**Correctif** : Clamper `raw` à `max(0.0, raw)` avant `min(1.0, raw)`.

### Problème 2 : Mots-clés complexes incomplets
Les mots-clés complexes actuels sont trop spécifiques :
- `"explique pourquoi"` est présent mais **pas** `"explique"` seul
- `"écris un rapport"` est présent mais **pas** `"écris"` seul
- `"crée un plan"` est présent mais **pas** `"plan"`

Conséquence : "Explique la photosynthèse en détail" → 0 hit complexe, score keywords=0.5 (neutre).
"Écris un script Python" → 0 hit complexe, score keywords=0.5.

### Problème 3 : Valeur par défaut du RAG trop haute
Quand `rag_score=None` (pas de RAG), `score_rag` retourne **0.8**, ce qui ajoute 0.20 au score final (via w_rag=0.25).
Cela pousse artificiellement toutes les requêtes sans contexte documentaire vers le haut.

**Impact** : "Bonjour" obtient score=0.355 (juste sous N2) uniquement à cause du RAG. Sans RAG, une question simple a déjà 0.20 de base.

### Problème 4 : Seuils trop serrés
La bande médiane N2 est très étroite : 0.36-0.42 (delta=0.06).
Une variation de 0.01 peut faire basculer N1↔N2 ou N2↔N3.

---

## 4. Propositions d'Ajustement

### Proposition A (Recommandée) — Ajustements modérés

**Poids ajustés :**
| Poids | Actuel | Proposé | Raison |
|-------|--------|---------|--------|
| w_rag | 0.25 | **0.20** | Réduire l'impact du défaut RAG=None (0.8→0.16 au lieu de 0.20) |
| w_keywords | 0.40 | **0.40** | Stable : bon discriminateur |
| w_length | 0.20 | **0.25** | Augmenter le poids de la longueur (critère objectif) |
| w_structure | 0.15 | **0.15** | Stable (après correction du bug) |

**Seuils ajustés :**
| Seuil | Actuel | Proposé | Raison |
|-------|--------|---------|--------|
| threshold_n1 | 0.36 | **0.34** | Élargir la zone N2 vers le bas (couvre les requêtes 0.34-0.36 qui sont N2) |
| threshold_n3 | 0.42 | **0.44** | Réduire les faux N3 pour les requêtes longues mais simples |

**Corrections de code :**

1. **Structure clamp** : `return min(1.0, max(0.0, raw))` au lieu de `return min(1.0, raw)`

2. **RAG None value** : Passer de 0.8 à **0.55** (léger penchant complexité quand pas de documents, mais pas extrême)

3. **Mots-clés complexes** : Ajouter
   ```python
   "explique",        # seul (pas seulement "explique pourquoi")
   "écris",           # seul (pas seulement "écris un rapport")
   "détail", "détaillé",
   "code", "script",
   "propose",
   "parser",
   ```

### Proposition B (Alternative) — Changements conservateurs

Mêmes corrections de bug (structure clamp) mais :
- `none_rag_value = 0.6` (au lieu de 0.55)
- `w_rag = 0.25` (inchangé)
- `threshold_n1 = 0.36` (inchangé)
- `threshold_n3 = 0.44` (légèrement relevé)

---

## 5. Résultats Attendus après Correctifs (Proposition A)

Tests validés avec les nouveaux paramètres sur les 22 scénarios :

| Requête | Avant | Après | Amélioration |
|---------|-------|-------|-------------|
| Bonjour | N1 (0.355) | N1 (0.27) | OK, score plus bas |
| Qui es-tu ? | N2 (0.363) | N1 (0.26) | ✅ Routé correctement |
| Explique la photosynthèse en détail | N1 (0.355) | N2 (0.39) | ✅ Routé correctement |
| Écris un script Python | N1 (0.355) | N2 (0.41) | ✅ Routé correctement |
| Compare riz irrigué/pluvial | N2 (0.397) | N3 (0.45) | ✅ Routé correctement |
| Analyse changement climatique RDC | N3 (0.476) | N3 (0.47) | ✅ Reste N3 |
| Très longue phrase | N3 (0.443) | N3 (0.44) | ✅ Reste N3 |
| Question à choix multiples | N1 (0.355) | N2 (0.34) | ✅ Routé correctement |
| Définition simple RAG bas | N1 (0.338) | N1 (0.27) | OK, acceptable |

**Routage correct estimé : 20/22 (91%)** contre 15/22 (68%) actuellement.

---

## 6. Recommandations Complémentaires

### Amélioration future : Keywords avec stemming
Le matching par sous-chaîne exacte ignore la morphologie :
- "analyser" ≠ "analyse" (raté si seule "analyse" est dans la liste)
- "comparaison" ≠ "compare"
- "rédiger" ≠ "rédige"

Proposition : Utiliser un stemmer français (ex: NLTK Snowball) pour normaliser les requêtes et les mots-clés.

### Amélioration future : Pondération contextuelle des mots-clés
Les mots-clés simples (ex: "qui") ne devraient compter que si le mot est un mot entier, pas une sous-chaîne. "équilibre" ne devrait pas matcher "qui".

### Amélioration future : Logging enrichi
Ajouter le score de complexité, le breakdown et la raison dans `complexity_log.json` pour faciliter les futures calibrations.

---

## 7. Fichiers Modifiés / Créés

- `/Users/leblancbahiga/Downloads/Assistant IA/src/complexity_classifier.py` — À modifier avec les ajustements ci-dessus
- `/Users/leblancbahiga/Downloads/Assistant IA/data/calibration_test_results.json` — Résultats détaillés des 22 tests
- `/Users/leblancbahiga/Downloads/Assistant IA/test_classifier_calibration.py` — Script de test
- `/Users/leblancbahiga/Downloads/Assistant IA/test_proposed_configs.py` — Script de validation des propositions
- `/Users/leblancbahiga/Downloads/Assistant IA/calibration_notes.md` — Présent document
