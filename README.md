# DAAR Projet 1 — Clone de `egrep` (prototype Python)
> Implémentation manuelle d’un moteur de recherche par expressions régulières fondé sur les automates finis.

Ce projet s’inscrit dans le cadre du cours **DAAR (Développement des Algorithmes d'Application Réticulaire)**.  
L’objectif est de reproduire le fonctionnement d’un moteur de recherche de motifs textuels semblable à `egrep`, en implémentant manuellement la chaîne théorique : **expression régulière → automate fini → minimisation → exécution**.

Le dépôt contient :
- un prototype Python fonctionnel (`src/egrep_clone.py`) implémentant la chaîne : RegEx → NFA (Thompson) → DFA (sous-ensembles) → minimisation (Hopcroft), et un raccourci KMP pour les motifs purement littéraux ;
- un gabarit de rapport (`report/rapport_template.md`) aligné avec l’énoncé ;
- un `Makefile` pour exécuter des tests locaux et générer une archive de rendu.

> ⚠️ Portée des ERE supportées : parenthèses `()`, alternative `|`, concaténation (implicite), étoile `*`, point `.`, lettres ASCII. Les autres métacaractères ne sont pas traités.

## Utilisation rapide

```bash
# Lister les lignes de FILE qui contiennent un motif PATTERN
python src/egrep_clone.py 'S(a|g|r)*on' path/to/file.txt
```

**Remarque :** Sur macOS (et sur certaines distributions Linux récentes), la commande `python` peut être absente.
Dans ce cas, utilisez `python3` à la place :
```bash
python3 src/egrep_clone.py 'S(a|g|r)*on' path/to/file.txt
```

## Validation simple vs `egrep`

```bash
# exemple rapide (fournissez votre propre fichier)
egrep 'ab*a' tests/demo.txt
python3 src/egrep_clone.py 'ab*a' tests/demo.txt

# sortie attendue
tests/demo.txt:3:abba
tests/demo.txt:4:aaaaa
tests/demo.txt:5:bababa
```

## Performance

- Si votre motif est **purement littéral**, le programme bascule sur **KMP** automatiquement.
- Pour les autres motifs, la recherche se fait via **DFA minimisé**, avec encapsulation du motif en `Σ* R Σ*` afin de faire un **substring match** comme `egrep`.
- Ce comportement garantit des performances quasi linéaires, similaires à celles d’`egrep`,  
  tout en conservant une implémentation purement algorithmique sans bibliothèque externe.

## Packaging du rendu

L’exécution principale se trouve dans `src/egrep_clone.py`, le Makefile propose des cibles pratiques :

```bash
make test          # lance une ou deux démos locales
make zip           # produit daar-projet-offline-VOTRE_NOM.zip (remplacez VOTRE_NOM)
```

Pour un rendu conforme à l’énoncé, ajoutez :
- un **binaire** (ex. via PyInstaller) *ou* fournissez votre propre implémentation C++/Java avec Makefile/Ant ;
- un **README** (ce fichier), le **rapport** (5–10 pages), le **code commenté**, un **Makefile**, et un répertoire d’**instances de test**.

## Pistes d’amélioration

- Ajouter `+`, `?`, quantificateurs bornés `{m,n}` si souhaité (hors périmètre minimal).
- Support de classes de caractères, ancrages, etc.
- Optimisations mémoire/temps (compression du DFA, alphabet restreint dynamique, etc.).
