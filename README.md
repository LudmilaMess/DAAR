# Projet DAAR — Clone d’`egrep` avec KMP, Boyer–Moore et Automates Finis

## Présentation générale

Ce projet a été réalisé dans le cadre du module **DAAR (Développement et Analyse d’Algorithmes de Recherche)**.  
Il a pour objectif de concevoir un moteur de recherche textuelle inspiré de la commande Unix `egrep`, capable d’analyser de grands fichiers texte en appliquant différentes stratégies de recherche :

- **Moteur à base d’automates finis** :  
  Construction d’un automate **NFA (Thompson)**, déterminisation en **DFA**, puis **minimisation (Hopcroft)**.  
- **Recherche optimisée pour motifs littéraux** :  
  Utilisation des algorithmes **Knuth–Morris–Pratt (KMP)** et **Boyer–Moore** (règle du mauvais caractère).

Le projet compare les performances, la complexité et la précision des trois approches.

---

## Architecture du projet

```
DAAR-Liu/
├── src/
│   └── egrep_clone.py          # Moteur principal : NFA, DFA, KMP, Boyer–Moore
├── tests/
│   ├── run_tests.py            # Script d’exécution et de mesure
│   ├── plot_results.py         # Génération des graphiques comparatifs
│   └── outputs/                # Résultats et graphiques produits
├── results.csv                 # Données expérimentales (export)
├── Rapport_DAAR_Ludmila_Messaoudi.docx  # Rapport complet du projet
├── Makefile                    # Automatisation (tests, nettoyage)
└── README.md                   # Ce fichier
```
- **src/** — moteur principal et algorithmes (NFA, DFA, KMP, Boyer–Moore)
- **tests/** — scripts d’évaluation et de visualisation des résultats
- **results.csv** — enregistrement automatique des mesures de performance
- **Makefile** — automatisation des tests et nettoyage du projet


---

## Exécution du moteur `egrep_clone.py`

### Recherche d’un motif
```bash
python3 src/egrep_clone.py "S(a|g|r)*on" tests/demo.txt
```
Sortie typique :
```
[DEBUG] NFA: 28 états, DFA: 7, DFA min: 4
tests/demo.txt:6:Sargon
tests/demo.txt:7:Saon
[DEBUG] Temps total: 0.000157s
```

### Mode littéral (KMP / Boyer–Moore)
```bash
python3 src/egrep_clone.py "Sargon" tests/demo.txt
```
Sortie :
```
[DEBUG] NFA: 0 états, DFA: 0, DFA min: 0
[DEBUG] Mode littéral détecté : comparaison KMP / Boyer–Moore (insensible à la casse)
tests/demo.txt:6:Sargon
[DEBUG] KMP: 0.003353s (1 lignes)
[DEBUG] Boyer–Moore: 0.000006s (1 lignes)
```

---

## Utilisation du Makefile

Le projet inclut un **Makefile** permettant d’exécuter rapidement les tests et de créer l’archive finale du rendu.

### Lancer tous les tests
```bash
make
```
### ou équivalent :
```bash
make test-all
```

### Exécuter uniquement les tests par automates
```bash
make test
```
### Exécuter uniquement les tests en mode KMP / Boyer–Moore
```bash
make test-kmp
```
### Nettoyer les fichiers temporaires
```bash
make clean
```
### Générer l’archive du rendu
```bash
make zip
```

Produit un fichier `daar-egrep-LiuYANG-LudmilaMessaoudi.zip` prêt à être soumis.

---

## Tests automatisés

Pour lancer tous les tests définis :
```bash
python3 tests/run_tests.py
```

Le script :
- Exécute plusieurs motifs sur différents fichiers (dont *Les Misérables*).  
- Mesure les **temps d’exécution**, le **nombre d’états** (NFA, DFA, DFAmin) et le **nombre de lignes correspondantes**.  
- Enregistre tout dans `results.csv`.  

### Exemple de résultats
| Fichier | Motif | NFA | DFA | DFAmin | Temps (s) | Lignes |
|:--------|:------|----:|----:|-------:|-----------:|-------:|
| demo.txt |ab*a | 14 | 4 | 3 | 0.023 | 3 |
| demo.txt |S(a\|g\|r)*on | 28 | 7 | 4 | 0.025 | 2 |
| 1.txt |the | 0 | 0 | 0 | 0.027 | 233 |
| 1.txt |independence | 0 | 0 | 0 | 0.024 | 8 |
| les_miserables.txt |(a\|b)*a | 18 | 3 | 2 | 0.401 | 1340 |
| les_miserables.txt |Monsieur le maire | 0 | 0 | 0 | 0.164 | 48 |
| les_miserables.txt |....a | 14 | 6 | 6 | 0.405 | 1340 |

---

## Visualisation des résultats

Pour générer les graphiques comparatifs :
```bash
python3 tests/plot_results.py
```

Trois fichiers sont créés dans `tests/outputs/` :
- **execution_comparative.png** → Temps d’exécution des trois approches (DFA / KMP / Boyer–Moore).  
- **states_comparison.png** → Comparaison du nombre d’états (NFA, DFA, DFAmin).  
- **matches_count.png** → Nombre de lignes trouvées par motif.

---

## Principaux algorithmes utilisés

| Algorithme | Rôle | Complexité | Description |
|-------------|------|-------------|--------------|
| **Thompson** | NFA | O(m) | Construction de l’automate non déterministe. |
| **Méthode des sous-ensembles** | DFA | O(2^m) | Déterminisation du NFA. |
| **Hopcroft** | Minimisation DFA | O(n log n) | Réduction du nombre d’états. |
| **KMP** | Recherche littérale | O(n + m) | Recherche linéaire avec préfixe/suffixe. |
| **Boyer–Moore** | Recherche littérale | O(n/m) en pratique | Sauts par mauvaise correspondance. |

---

## Analyse synthétique

- **Boyer–Moore** est le plus rapide pour les motifs simples sur de grands fichiers.  
- **KMP** offre une performance linéaire et régulière.  
- **DFA** est plus coûteux à compiler mais constant en temps d’exécution une fois construit.  
- La **minimisation** réduit jusqu’à 70 % des états.  

**En conclusion**, ce projet illustre la complémentarité entre les approches **formelles (automates)** et **algorithmiques (KMP, Boyer–Moore)** pour la recherche efficace de motifs dans de grands corpus textuels.



---

## Références

- Hopcroft, J. E., Motwani, R., & Ullman, J. D. *Introduction to Automata Theory, Languages, and Computation.*  
- Knuth, D. E., Morris, J. H., & Pratt, V. R. *Fast Pattern Matching in Strings.* (1977)  
- Boyer, R. S., & Moore, J. S. *A Fast String Searching Algorithm.* (1977)  
- Cours DAAR – Université de Haute-Alsace, 2024–2025.  
- Documentation Python 3.14 — https://docs.python.org/3/

---

## Auteur

- **Ludmila Messaoudi, Liu YANG**  
Sorbonne Université - Master 2 Informatique Parcours Science et Technologie du Logiciel en alternance 

**Date : Octobre 2025**
Projet réalisé dans le cadre du module **DAAR (Développement et Analyse d’Algorithmes de Recherche)**.
