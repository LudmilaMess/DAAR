# Rapport — Projet 1 : Clone d’`egrep` (offline, desktop)

**Auteurs :** NOM1, NOM2  
**UE :** DAAR — Projet algorithmique  
**Date :** 2025-10-05 (deadline de dépôt)

## 1. Problème & structures de données

- **Problème** : Recherche d’un motif (RegEx, norme ERE restreinte) dans un **fichier texte**, ligne par ligne, avec un résultat identique à `egrep`.
- **Structures** :
  - **AST/Polonaise** du motif (via shunting-yard) ;
  - **NFA** (Thompson) + transitions ε et `ANY` ;
  - **DFA** (méthode des sous-ensembles) ;
  - **DFA minimisé** (Hopcroft) ;
  - **KMP** pour les motifs purement littéraux.

## 2. Algorithmes connus (littérature)

- **Thompson (1968)** pour construire un NFA à partir d’une RegEx.
- **Méthode des sous-ensembles** pour déterminer un DFA.
- **Minimisation de DFA** (**Hopcroft**) pour réduire le nombre d’états.
- **KMP** (Knuth–Morris–Pratt) pour la correspondance de chaînes lorsque le motif est un mot simple.

## 3. Choix & améliorations

- Encapsulation du motif en **Σ\* R Σ\*** dans le NFA pour un **substring match**.
- Alphabet réduit aux **256 octets** ; l’opérateur `.` et les boucles `Σ*` sont modélisés par un **symbole ANY**.
- Basculement automatique **KMP** si le motif ne contient pas de méta-caractères.

## 4. Jeux de tests

- Méthode : créer des corpus à partir de textes **Gutenberg** (romans, poésie) ; générer des motifs simples/complexes ; varier la taille du fichier.
- Exemples : alternances, itérations, parenthèses imbriquées, etc.

## 5. Benchmarks

- Mesures : temps total de recherche, nombre d’états NFA/DFA, mémoire.
- Visualisation : courbes, barres (moyenne + écart-type), histogrammes de fréquences.
- Comparaison : `egrep` vs notre implémentation (et KMP pour motifs littéraux).

## 6. Discussion

- Analyse des résultats, cas défavorables (explosion du DFA), limites, idées d’optimisation.

## 7. Conclusion & perspectives

- Bilan et pistes futures.

## Annexes

- Guide d’utilisation.
- Détails sur la grammaire supportée.
- Preuves de correction/terminaison (esquisse).
