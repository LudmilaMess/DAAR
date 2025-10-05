import matplotlib.pyplot as plt
import os

# === Dossier de sortie (tous les graphiques seront enregistrés ici) ===
output_dir = "tests/outputs"
os.makedirs(output_dir, exist_ok=True)

# ==== Données issues du rapport ====
motifs = ['ab*a', '(ab)*c', '(a|b)*a', '....a', 'S(a|g|r)*on']
nfa = [14, 14, 18, 14, 28]
dfa = [7, 7, 5, 7, 13]
dfa_min = [3, 2, 2, 6, 4]
temps = [0.415, 0.447, 0.445, 0.447, 0.414]

# ==== Figure 1 : Évolution du nombre d’états ====
plt.figure(figsize=(8, 5))
plt.plot(motifs, nfa, 'o-', label='NFA')
plt.plot(motifs, dfa, 'o-', label='DFA')
plt.plot(motifs, dfa_min, 'o-', label='DFA minimisé')
plt.title("Évolution du nombre d’états selon le motif")
plt.xlabel("Motif")
plt.ylabel("Nombre d’états")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "states_comparison.png"), dpi=300)
plt.show()

# ==== Figure 2 : Temps d’exécution ====
plt.figure(figsize=(8, 5))
plt.bar(motifs, temps, color='skyblue', edgecolor='black')
plt.title("Temps d’exécution par motif")
plt.xlabel("Motif")
plt.ylabel("Temps (secondes)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "execution_time.png"), dpi=300)
plt.show()

print(f" Graphiques générés dans le dossier : {output_dir}/")