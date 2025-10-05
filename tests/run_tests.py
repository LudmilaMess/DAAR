#!/usr/bin/env python3
import subprocess
import time
import re
import os

# === Dossier de sortie ===
output_dir = "tests/outputs"
os.makedirs(output_dir, exist_ok=True)

# === Liste des motifs à tester ===
motifs = [
    "ab*a",
    "(ab)*c",
    "(a|b)*a",
    "....a",
    "S(a|g|r)*on"
]

# === En-tête du tableau ===
print(f"{'Motif':<25} {'NFA':>5} {'DFA':>6} {'DFAmin':>8} {'Temps(s)':>10} {'Lignes':>10}")
print("-" * 65)

# === Exécution des tests ===
for motif in motifs:
    # Nettoyage du nom de fichier (pas de caractères spéciaux)
    nom = motif.replace('*', 'STAR').replace('(', '_').replace(')', '_').replace('|', 'OR').replace('.', 'DOT')
    fichier_sortie = os.path.join(output_dir, f"out_{nom}.txt")

    # Mesure du temps d’exécution
    t0 = time.time()
    process = subprocess.run(
        ["python3", "src/egrep_clone.py", motif, "tests/les_miserables.txt"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    temps = time.time() - t0

    # Sauvegarde du résultat dans tests/outputs/
    with open(fichier_sortie, "w") as f:
        f.write(process.stdout)

    # Compte des lignes trouvées
    lignes = len(process.stdout.strip().splitlines())

    # Extraction des informations NFA/DFA depuis la sortie d’erreur
    debug_output = process.stderr
    match = re.search(r"NFA:\s*(\d+)\s+états,\s*DFA:\s*(\d+),\s*DFA min:\s*(\d+)", debug_output)
    if match:
        nfa, dfa, dfa_min = match.groups()
    else:
        nfa, dfa, dfa_min = "?", "?", "?"

    # Affichage formaté des résultats
    print(f"{motif:<25} {nfa:>5} {dfa:>6} {dfa_min:>8} {temps:10.3f} {lignes:10}")

print("\n Tous les résultats ont été enregistrés dans le dossier 'tests/outputs/'")
