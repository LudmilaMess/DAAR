# DAAR Projet 1 — Clone de `egrep` (prototype Python)

> **Objet** : Recherche de motif par expression régulière (sous-ensemble ERE) sur un fichier texte, ligne par ligne, avec comparaison aux résultats d’`egrep`.

Ce dépôt vous fournit :
- un **prototype Python** fonctionnel (`src/egrep_clone.py`) implémentant la chaîne : RegEx → **NFA (Thompson)** → **DFA (sous-ensembles)** → **minimisation (Hopcroft)**, et un **raccourci KMP** pour les motifs purement littéraux ;
- un **gabarit de rapport** (`report/rapport_template.md`) aligné avec l’énoncé ;
- un **Makefile** pour exécuter des tests locaux et cibler une archive de rendu.

> ⚠️ Portée des ERE supportées : parenthèses `()`, alternative `|`, concaténation (implicite), étoile `*`, point `.`, lettres ASCII. Les autres métacaractères ne sont pas traités.

## Utilisation rapide

```bash
# Lister les lignes de FILE qui contiennent un motif PATTERN
python3 src/egrep_clone.py 'S(a|g|r)+on' path/to/file.txt
```

**Remarque :** pour l’instant, l’opérateur `+` n’est pas supporté ; remplacez-le par `aa*` si nécessaire (ex. `x+` ↦ `xx*`).

## Validation simple vs `egrep`

```bash
# exemple rapide (fournissez votre propre fichier)
egrep 'ab*a' tests/demo.txt
python3 src/egrep_clone.py 'ab*a' tests/demo.txt
```

## Performance

- Si votre motif est **purement littéral**, le programme bascule sur **KMP** automatiquement.
- Pour les autres motifs, la recherche se fait via **DFA minimisé**, avec encapsulation du motif en `Σ* R Σ*` afin de faire un **substring match** comme `egrep`.

## Packaging du rendu

Le Makefile propose des cibles pratiques :

```bash
make test          # lance une ou deux démos locales
make zip           # produit daar-projet-offline-VOTRE_NOM.zip (à éditer)
```

Pour un rendu conforme à l’énoncé, ajoutez :
- un **binaire** (ex. via PyInstaller) *ou* fournissez votre propre implémentation C++/Java avec Makefile/Ant ;
- un **README** (ce fichier), le **rapport** (5–10 pages), le **code commenté**, un **Makefile**, et un répertoire d’**instances de test**.

## Pistes d’amélioration

- Ajouter `+`, `?`, quantificateurs bornés `{m,n}` si souhaité (hors périmètre minimal).
- Support de classes de caractères, ancrages, etc.
- Optimisations mémoire/temps (compression du DFA, alphabet restreint dynamique, etc.).
