#!/usr/bin/env python3
import subprocess
import time
import re
import os
import csv

# === Dossier de sortie ===
output_dir = "tests/outputs"
os.makedirs(output_dir, exist_ok=True)

# === Liste des (motif, fichier) à tester ===
tests = [
    ("ab*a", "demo.txt"),
    ("S(a|g|r)*on", "demo.txt"),
    ("the", "1.txt"),
    ("independence", "1.txt"),
    ("(a|b)*a", "les_miserables.txt"),
    ("Monsieur le maire", "les_miserables.txt"),
    ("....a", "les_miserables.txt"),
]

# === CSV de sortie pour plot_results.py ===
csv_file = "results.csv"

# === En-tête de la console ===
print(f"{'Fichier':<22} {'Motif':<25} {'NFA':>6} {'DFA':>6} {'DFAmin':>8} {'Temps(s)':>10} {'Lignes':>10}")
print("-" * 90)

rows = []

# === Exécution des tests ===
for motif, fichier in tests:
    nom = motif.replace('*', 'STAR').replace('(', '_').replace(')', '_').replace('|', 'OR').replace('.', 'DOT').replace(' ', '_')
    fichier_sortie = os.path.join(output_dir, f"out_{os.path.basename(fichier)}_{nom}.txt")

    t0 = time.time()
    process = subprocess.run(
    ["python3", "src/egrep_clone.py", motif, os.path.join("tests", fichier)],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
    )
    temps = time.time() - t0

    # Sauvegarde de la sortie brute
    with open(fichier_sortie, "w") as f:
        f.write(process.stdout)

    # Comptage des lignes correspondantes
    lignes = 0
    for l in process.stdout.splitlines():
        m = re.match(r'^(.+?):\d+:', l)
        if m:
            printed_path = m.group(1)
            if os.path.basename(printed_path) == fichier:
                lignes += 1


    # --- Fusionner stdout et stderr pour capturer tous les messages de débogage ---
    # Certains motifs (par ex. en mode littéral) impriment les infos sur stdout au lieu de stderr.
    # L’expression régulière suivante accepte les formats avec ':' ou '=' et ignore la casse.
    debug_output = process.stderr + process.stdout
    match = re.search(
        r"NFA\s*[:=]\s*(\d+).*?DFA\s*[:=]\s*(\d+).*?(?:DFA\s*min|DFAmin)\s*[:=]\s*(\d+)",
        debug_output,
        re.IGNORECASE
    )

    if match:
        nfa, dfa, dfa_min = match.groups()
    else:
        nfa, dfa, dfa_min = "?", "?", "?"

    # Affichage console
    print(f"{fichier:<22} {motif:<25} {nfa:>6} {dfa:>6} {dfa_min:>8} {temps:10.3f} {lignes:10}")

    # Enregistrement CSV
    rows.append({
        "file": fichier,
        "pattern": motif,
        "nfa": nfa,
        "dfa": dfa,
        "dfa_min": dfa_min,
        "time": round(temps, 3),
        "matches": lignes,
    })

# === Écriture du fichier CSV (pour les graphes) ===
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "pattern", "nfa", "dfa", "dfa_min", "time", "matches"])
    writer.writeheader()
    writer.writerows(rows)

print("\n Tous les résultats ont été enregistrés dans le dossier 'tests/outputs/' et dans 'results.csv'")