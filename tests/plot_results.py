#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import os

# === Dossier de sortie ===
output_dir = "tests/outputs"
os.makedirs(output_dir, exist_ok=True)

# === Lecture du fichier CSV généré par run_tests.py ===
csv_path = os.path.join(output_dir, "results.csv")
df = pd.read_csv(csv_path)

# === Filtrage : ne garder que les lignes avec valeurs numériques ===
df = df[df["nfa"] != "?"]

# === Conversion des colonnes en numériques ===
df["nfa"] = pd.to_numeric(df["nfa"], errors="coerce")
df["dfa"] = pd.to_numeric(df["dfa"], errors="coerce")
df["dfa_min"] = pd.to_numeric(df["dfa_min"], errors="coerce")
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["matches"] = pd.to_numeric(df["matches"], errors="coerce")

# === Figure 1 : Nombre d’états ===
plt.figure(figsize=(8, 5))
plt.plot(df["pattern"], df["nfa"], 'o-', label="NFA")
plt.plot(df["pattern"], df["dfa"], 'o-', label="DFA")
plt.plot(df["pattern"], df["dfa_min"], 'o-', label="DFA minimisé")
plt.title("Évolution du nombre d’états selon le motif")
plt.xlabel("Motif")
plt.ylabel("Nombre d’états")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "states_comparison.png"), dpi=300)

# === Figure 2 : Temps d’exécution ===
plt.figure(figsize=(8, 5))
plt.bar(df["pattern"], df["time"], color='skyblue', edgecolor='black')
plt.title("Temps d’exécution par motif")
plt.xlabel("Motif")
plt.ylabel("Temps (secondes)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "execution_time.png"), dpi=300)

# === Figure 3 : Nombre de lignes correspondantes ===
plt.figure(figsize=(8, 5))
plt.bar(df["pattern"], df["matches"], color='lightgreen', edgecolor='black')
plt.title("Nombre de lignes correspondantes par motif")
plt.xlabel("Motif")
plt.ylabel("Nombre de lignes trouvées")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "matches_count.png"), dpi=300)
plt.show()

print(f"\n Graphiques générés dans le dossier : {output_dir}/")