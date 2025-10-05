#!/usr/bin/env python3
"""
egrep_clone.py — DAAR Projet 1 : recherche hors ligne par expressions régulières (sous-ensemble de ERE)
Opérateurs pris en charge : (), |, concaténation (implicite), *, ., lettres ASCII.

Utilisation :
  python3 egrep_clone.py MOTIF FICHIER

Ce prototype compile le motif (PATTERN) en un NFA (méthode de Thompson),
le convertit en DFA (construction par sous-ensemble),
puis le minimise via l’algorithme de Hopcroft,
et recherche chaque ligne du fichier pour une correspondance de sous-chaîne (Σ* R Σ*).
Si le motif est un littéral simple (lettres ASCII uniquement), il utilise KMP.
"""

import sys
from collections import defaultdict, deque

ANY = None  # symbole générique pour '.' et '.*'

# -------------------- Tokenisation et algorithme du "Shunting-yard" --------------------

def tokenize(regex: str):
    """Analyse lexicale : transforme une chaîne regex en liste de symboles."""
    tokens = []
    i = 0
    while i < len(regex):
        c = regex[i]
        if c in '()*|.':
            # caractères spéciaux (opérateurs ou parenthèses)
            tokens.append(c)
            i += 1
        elif c == '\\' and i + 1 < len(regex):
            # gestion du caractère d’échappement : on ajoute le suivant comme littéral
            tokens.append(regex[i+1])
            i += 2
        elif 0x20 <= ord(c) <= 0x7E:  # caractères ASCII imprimables interprétés comme symboles
            tokens.append(c)
            i += 1
        else:
            # caractère non pris en charge
            raise ValueError(f"Caractère non supporté dans le motif : U+{ord(c):04X}")
    return tokens

def insert_concat(tokens):
    """Insère explicitement l’opérateur de concaténation '·' entre symboles contigus."""
    out = []
    for i, t in enumerate(tokens):
        out.append(t)
        if i + 1 < len(tokens):
            t1, t2 = tokens[i], tokens[i+1]
            # Si t1 est un symbole, ')', ou '*', et t2 est un symbole, '(' ou '.',
            # on insère l’opérateur de concaténation '·' pour rendre la structure explicite.
            if ((is_symbol(t1) or t1 in (')','*','.')) and
                (is_symbol(t2) or t2 in ('(','.',))):
                out.append('·')  # opérateur de concaténation explicite
    return out

def is_symbol(t):
    """Détermine si un token est un symbole de base (non opérateur)."""
    return t not in {'(',')','|','*','·','.'}

def to_postfix(tokens):
    """
    Convertit une expression régulière en notation postfixée
    à l’aide de l’algorithme du shunting-yard (Dijkstra).
    """
    prec = {'*':3, '·':2, '|':1}  # hiérarchie de précédence des opérateurs
    out = []
    stack = []
    for t in tokens:
        if is_symbol(t) or t == '.':
            # symbole simple → directement en sortie
            out.append(t)
        elif t in ('|','·','*'):
            if t == '*':
                # opérateur unaire postfixé — on dépile les autres '*' de même précédence
                while stack and stack[-1] == '*':
                    out.append(stack.pop())
                stack.append(t)
            else:
                # opérateurs binaires : on dépile ceux de précédence ≥
                while stack and stack[-1] != '(' and prec.get(stack[-1],0) >= prec[t]:
                    out.append(stack.pop())
                stack.append(t)
        elif t == '(':
            # parenthèse ouvrante : on empile
            stack.append(t)
        elif t == ')':
            # parenthèse fermante : on dépile jusqu’à '('
            while stack and stack[-1] != '(':
                out.append(stack.pop())
            if not stack:
                raise ValueError("Parenthèses non appariées")
            stack.pop()
        else:
            raise ValueError(f"Symbole inconnu : {t}")
    # On vide la pile restante
    while stack:
        if stack[-1] in '()':
            raise ValueError("Parenthèses non appariées")
        out.append(stack.pop())
    return out

# -------------------- NFA de Thompson --------------------

class NFA:
    """Représente un automate fini non déterministe (AFN/NFA)."""
    __slots__ = ('start', 'accepts', 'eps', 'trans', 'states')
    def __init__(self):
        self.start = 0                           # état initial
        self.accepts = set()                     # ensemble des états d’acceptation
        self.eps = defaultdict(set)              # transitions ε : état -> {états cibles}
        self.trans = defaultdict(lambda: defaultdict(set))  # transitions symboliques : état -> {symbole -> {états cibles}}
        self.states = 0                          # nombre total d’états

def new_state(nfa: NFA):
    """Crée un nouvel état et renvoie son identifiant."""
    s = nfa.states
    nfa.states += 1
    return s

def frag_symbol(symbol):
    """Construit un fragment d’AFN pour un symbole unique (ou pour '.')."""
    n = NFA()
    s = new_state(n)
    f = new_state(n)
    if symbol == '.':
        n.trans[s][ANY].add(f)  # le point '.' représente un joker (ANY)
    else:
        n.trans[s][symbol].add(f)
    n.start = s
    n.accepts = {f}
    return n

def frag_concat(a: NFA, b: NFA):
    """Concaténation de deux fragments d’AFN (a suivi de b)."""
    off = a.states
    n = NFA()
    n.states = a.states + b.states
    # Copie des transitions et ε-transitions du fragment a
    for u in range(a.states):
        for sym, vs in a.trans[u].items():
            n.trans[u][sym] |= {v for v in vs}
        n.eps[u] |= set(a.eps[u])
    # Copie du fragment b en décalant ses états
    for u in range(b.states):
        for sym, vs in b.trans[u].items():
            n.trans[u+off][sym] |= {v+off for v in vs}
        n.eps[u+off] |= {v+off for v in b.eps[u]}
    # Connexion entre les états d’acceptation de a et le début de b via ε
    for aacc in a.accepts:
        n.eps[aacc].add(b.start + off)
    n.start = a.start
    n.accepts = {v+off for v in b.accepts}
    return n

def frag_union(a: NFA, b: NFA):
    """Union (|) de deux AFN : crée un nouveau départ et une nouvelle fin."""
    off = a.states
    n = NFA()
    n.states = a.states + b.states + 2
    s = new_state(n)  # nouveau départ
    f = new_state(n)  # nouvel état d’acceptation global
    # Copie du fragment a
    for u in range(a.states):
        for sym, vs in a.trans[u].items():
            n.trans[u][sym] |= set(vs)
        n.eps[u] |= set(a.eps[u])
    # Copie du fragment b (décalé)
    for u in range(b.states):
        for sym, vs in b.trans[u].items():
            n.trans[u+off][sym] |= {v+off for v in vs}
        n.eps[u+off] |= {v+off for v in b.eps[u]}
    # Connexions par ε entre le nouveau départ et les deux sous-automates
    n.eps[s].add(a.start)
    n.eps[s].add(b.start + off)
    # Connexion des anciens états d’acceptation vers le nouvel état final
    for aacc in a.accepts:
        n.eps[aacc].add(f)
    for bacc in b.accepts:
        n.eps[bacc+off].add(f)
    n.start = s
    n.accepts = {f}
    return n

def frag_star(a: NFA):
    """Fermeture de Kleene (*) : répète le fragment a zéro ou plusieurs fois."""
    n = NFA()
    n.states = a.states + 2
    s = new_state(n)   # nouveau départ
    f = new_state(n)   # nouveau état final
    # Copie du fragment a
    for u in range(a.states):
        for sym, vs in a.trans[u].items():
            n.trans[u][sym] |= set(vs)
        n.eps[u] |= set(a.eps[u])
    # Connexions ε : boucle permettant répétition
    n.eps[s].add(a.start)
    n.eps[s].add(f)
    for acc in a.accepts:
        n.eps[acc].add(f)
        n.eps[acc].add(a.start)
    n.start = s
    n.accepts = {f}
    return n

def build_nfa_from_postfix(postfix):
    """Construit un AFN complet à partir d’une expression postfixée."""
    stack = []
    for t in postfix:
        if is_symbol(t) or t == '.':
            stack.append(frag_symbol(t))
        elif t == '·':
            b = stack.pop(); a = stack.pop()
            stack.append(frag_concat(a,b))
        elif t == '|':
            b = stack.pop(); a = stack.pop()
            stack.append(frag_union(a,b))
        elif t == '*':
            a = stack.pop()
            stack.append(frag_star(a))
        else:
            raise ValueError(f"Symbole postfixé inconnu {t}")
    if len(stack) != 1:
        raise ValueError("Expression postfixée invalide")
    return stack[0]

def add_search_wrappers(nfa: NFA):
    """
    Enveloppe l’AFN avec Σ* (avant et après)
    pour permettre la recherche de sous-chaînes (Σ* R Σ*).
    """
    wrapped = NFA()
    wrapped.states = nfa.states + 2
    s = new_state(wrapped)  # nouveau départ
    # Copie des transitions du NFA d’origine
    for u in range(nfa.states):
        for sym, vs in nfa.trans[u].items():
            wrapped.trans[u][sym] |= set(vs)
        wrapped.eps[u] |= set(nfa.eps[u])
    # Boucle sur ANY au nouveau départ : Σ*
    wrapped.trans[s][ANY].add(s)
    wrapped.eps[s].add(nfa.start)
    # Création d’un nouvel état d’acceptation et ajout de Σ* après
    f = new_state(wrapped)
    for acc in nfa.accepts:
        wrapped.eps[acc].add(f)
    wrapped.trans[f][ANY].add(f)  # absorbe tout après le motif
    wrapped.start = s
    wrapped.accepts = {f}
    return wrapped

# -------------------- Construction par sous-ensemble (NFA -> DFA) --------------------

class DFA:
    """Représente un automate fini déterministe (AFD/DFA) obtenu depuis l’AFN par la méthode des sous-ensembles."""
    __slots__ = ('start', 'accepts', 'trans', 'states_map', 'states_rev')
    def __init__(self):
        self.start = 0
        self.accepts = set()                       # états d’acceptation DFA (indices d’états DFA)
        self.trans = defaultdict(dict)             # transitions déterministes : état_DFA -> {octet -> état_DFA}
        self.states_map = {}                       # mapping : frozenset(états_NFA) -> id_état_DFA
        self.states_rev = []                       # reverse mapping : id_état_DFA -> frozenset(états_NFA)

def epsilon_closure(nfa: NFA, states):
    """
    ε-fermeture : à partir d’un ensemble d’états NFA `states`, on ajoute
    tous les états accessibles uniquement via des transitions ε.
    Renvoie un set d’états NFA.
    """
    stack = list(states)
    seen = set(states)
    while stack:
        u = stack.pop()
        for v in nfa.eps[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen

def move(nfa: NFA, states, byte_val):
    """
    Calcule l’ensemble des états NFA atteignables depuis `states`
    en consommant un octet `byte_val` (prise en charge du joker ANY pour '.').
    """
    nxt = set()
    ch = chr(byte_val)                # conversion octet -> caractère pour indexer nfa.trans[u][ch]
    for u in states:
        # Transitions joker ('.') : ANY signifie « n’importe quel symbole »
        if ANY in nfa.trans[u]:
            nxt |= nfa.trans[u][ANY]
        # Transition sur le caractère exact
        if ch in nfa.trans[u]:
            nxt |= nfa.trans[u][ch]
    return nxt

def determinize(nfa: NFA):
    """
    Déterminisation (méthode des sous-ensembles) :
    - Un état DFA = un ensemble (frozenset) d’états NFA.
    - L’état initial DFA = ε-fermeture({nfa.start}).
    - Pour chaque état DFA S et chaque octet b ∈ [0..255] :
        T = ε-fermeture(move(NFA, S, b)).
        On crée/identifie l’état DFA correspondant à T et on ajoute la transition S --b--> T.
    - Un état DFA est acceptant ssi il contient au moins un état acceptant du NFA.
    """
    dfa = DFA()

    # 1) État initial du DFA = ε-fermeture du départ NFA
    start_set = frozenset(epsilon_closure(nfa, {nfa.start}))
    dfa.states_map[start_set] = 0
    dfa.states_rev.append(start_set)
    if any(s in nfa.accepts for s in start_set):
        dfa.accepts.add(0)

    # 2) Parcours en largeur des états DFA « atteignables »
    q = deque([0])
    while q:
        s_id = q.popleft()
        S = dfa.states_rev[s_id]  # S = frozenset d’états NFA (l’état DFA courant)

        # 3) Pour chaque octet possible (alphabet : 0..255), on calcule la transition
        for b in range(256):
            M = move(nfa, S, b)   # Move(S, b) : où peut-on aller depuis S en lisant l’octet b (sans ε)
            if not M:
                continue          # aucune cible → pas de transition sortante pour b (DFA partiel)
            T = frozenset(epsilon_closure(nfa, M))  # on complète avec les ε-transitions → T

            # 4) Si T est un « nouvel » ensemble d’états NFA, on l’enregistre comme nouvel état DFA
            if T not in dfa.states_map:
                dfa.states_map[T] = len(dfa.states_rev)
                dfa.states_rev.append(T)
                # 5) État acceptant si T contient un état acceptant NFA
                if any(t in nfa.accepts for t in T):
                    dfa.accepts.add(dfa.states_map[T])
                q.append(dfa.states_map[T])

            # 6) Ajout de la transition déterministe s_id --b--> id(T)
            dfa.trans[s_id][b] = dfa.states_map[T]

    return dfa

# -------------------- Minimisation de Hopcroft --------------------

def hopcroft_minimize(dfa: DFA):
    """
    Algorithme de Hopcroft : minimise un automate déterministe (DFA).
    Principe :
      - On partitionne les états en deux ensembles : acceptants / non-acceptants.
      - On raffine la partition tant que certains blocs peuvent être divisés
        par leur comportement sur les transitions (préimages selon les symboles).
      - Chaque bloc final devient un état du DFA minimal.
    """
    # Initialisation de la partition P avec les états acceptants et non-acceptants
    all_states = set(range(len(dfa.states_rev)))  # ensemble de tous les états
    A = set(dfa.accepts)                          # états acceptants
    NA = all_states - A                           # états non-acceptants
    P = [A, NA] if NA else [A]                    # partition initiale
    # Le « travail » W contient les blocs à raffiner (on choisit le plus petit pour efficacité)
    W = [A] if len(A) <= len(NA) else [NA]

    # Pré-calcul : pour chaque symbole b ∈ [0..255], inverse des transitions (préimages)
    # inv_trans[b][v] = {u | u --b--> v}
    inv_trans = [defaultdict(set) for _ in range(256)]
    for u, edges in dfa.trans.items():
        for b, v in edges.items():
            inv_trans[b][v].add(u)

    # Raffinement de la partition
    while W:
        Aset = W.pop()  # on extrait un bloc de travail
        for b in range(256):  # on traite toutes les lettres de l’alphabet
            # X = ensemble des états ayant une transition sur b vers un état de Aset
            X = set()
            for q in Aset:
                X |= inv_trans[b].get(q, set())
            if not X:
                continue
            newP = []
            # On parcourt chaque bloc Y dans la partition P
            for Y in P:
                inter = Y & X   # sous-ensemble de Y qui « va » dans Aset via b
                diff = Y - X    # sous-ensemble de Y qui « ne va pas » dans Aset via b
                # Si les deux ne sont pas vides, Y doit être divisé
                if inter and diff:
                    newP.extend([inter, diff])
                    # Gestion du travail W
                    if Y in W:
                        # Si Y était déjà dans W, on le remplace par ses deux sous-blocs
                        W.remove(Y)
                        W.extend([inter, diff])
                    else:
                        # Sinon, on ajoute le plus petit des deux sous-blocs
                        W.append(inter if len(inter) <= len(diff) else diff)
                else:
                    newP.append(Y)
            P = newP  # mise à jour de la partition

    # -------------------- Construction du DFA minimal --------------------
    rep = {}
    # Pour chaque bloc, on choisit le plus petit indice comme représentant
    for block in P:
        r = min(block)
        for s in block:
            rep[s] = r

    # Nouveaux états : triés selon les représentants uniques
    new_states = sorted({rep[s] for s in all_states})
    # Mapping : ancien représentant -> nouvel identifiant (0,1,2,...)
    new_id = {s: i for i, s in enumerate(new_states)}

    # Création du nouvel automate minimal
    mdfa = DFA()
    # Nouvel état de départ = image du représentant de l’état initial
    mdfa.start = new_id[rep[dfa.start]]
    # Prépare la liste des états (placeholders)
    for s_old in new_states:
        mdfa.states_rev.append(set())
    # Copie des transitions en suivant la correspondance de représentants
    for u, edges in dfa.trans.items():
        u2 = new_id[rep[u]]
        for b, v in edges.items():
            v2 = new_id[rep[v]]
            mdfa.trans[u2][b] = v2
    # Copie des états d’acceptation
    for a in dfa.accepts:
        mdfa.accepts.add(new_id[rep[a]])
    return mdfa

# -------------------- KMP (chemin rapide pour les motifs littéraux) --------------------

def is_literal(regex: str):
    """
    Vérifie si le motif est un littéral pur (sans métacaractères ni opérateurs).
    Si le motif ne contient que des lettres ASCII ordinaires, on pourra utiliser KMP.
    """
    for c in regex:
        if c in '()*|.' or c == '\\':
            return False
    return True

def kmp_build(pattern: bytes):
    """
    Construit le tableau de préfixes (pi) pour l’algorithme de Knuth–Morris–Pratt (KMP).
    - pi[i] = longueur du plus long préfixe propre de pattern[0..i] qui est aussi un suffixe.
    - Permet, en cas d’échec de correspondance, de ne pas recommencer depuis le début.
    """
    pi = [0] * len(pattern)
    j = 0  # longueur du préfixe courant correspondu
    for i in range(1, len(pattern)):
        # Si le prochain caractère ne correspond pas, on revient au plus long préfixe valide
        while j and pattern[i] != pattern[j]:
            j = pi[j-1]
        # Si le caractère correspond, on étend le préfixe commun
        if pattern[i] == pattern[j]:
            j += 1
            pi[i] = j
    return pi

def kmp_search(pattern: bytes, text: bytes):
    """
    Recherche du motif `pattern` dans le texte `text` en utilisant l’algorithme KMP.
    Avantage : complexité O(n), sans retour arrière.
    """
    if not pattern:
        return True  # motif vide → correspond toujours
    pi = kmp_build(pattern)
    j = 0  # indice dans le motif
    for i in range(len(text)):
        # En cas de désaccord, on recule selon la table pi
        while j and text[i] != pattern[j]:
            j = pi[j-1]
        # Si les caractères correspondent
        if text[i] == pattern[j]:
            j += 1
            # Si on a parcouru tout le motif → correspondance trouvée
            if j == len(pattern):
                return True
    return False

# -------------------- Moteur principal : compilation et recherche --------------------

class Engine:
    """
    Classe principale du moteur :
    - Si le motif est un littéral pur → utilise KMP (chemin rapide)
    - Sinon → compile en NFA (Thompson) → DFA (déterminisation) → minimisation (Hopcroft)
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.mode = 'regex'
        if is_literal(pattern):
            # Mode KMP : cas optimisé pour motifs simples
            self.mode = 'kmp'
            # Conversion du motif en octets (latin-1)
            self.literal = pattern.encode('latin-1', 'ignore')
        else:
            # Analyse lexicale et parsing
            tokens = tokenize(pattern)
            tokens = insert_concat(tokens)
            postfix = to_postfix(tokens)

            # Compilation : NFA → DFA → DFA minimal
            nfa = build_nfa_from_postfix(postfix)
            nfa = add_search_wrappers(nfa)
            dfa = determinize(nfa)
            self.dfa = hopcroft_minimize(dfa)
            print(f"[DEBUG] NFA: {nfa.states} états, DFA: {len(dfa.states_rev)}, DFA min: {len(self.dfa.states_rev)}", file=sys.stderr)

    def match_line(self, line: bytes) -> bool:
        """
        Recherche si une ligne (en bytes) correspond au motif.
        Si mode KMP : exécution directe de l’algorithme KMP.
        Sinon : parcours de l’automate DFA.
        """
        if self.mode == 'kmp':
            return kmp_search(self.literal, line)

        # -------------------- Parcours du DFA --------------------
        s = self.dfa.start
        for b in line:
            # Lecture d’un octet, suivre la transition si elle existe
            s = self.dfa.trans.get(s, {}).get(b, None)
            if s is None:
                # transition morte : pas de sortie pour ce caractère
                # → on redémarre depuis l’état initial (Σ* R Σ*)
                s = self.dfa.start
                s = self.dfa.trans.get(s, {}).get(b, s)

            # Note : on pourrait quitter dès qu’un état acceptant est atteint,
            # mais on choisit de continuer jusqu’à la fin pour fiabilité.
        return s in self.dfa.accepts


def grep_file(pattern: str, path: str):
    """
    Fonction utilitaire qui parcourt un fichier et affiche
    toutes les lignes correspondant au motif (comme `egrep`).
    """
    eng = Engine(pattern)
    count = 0
    with open(path, 'rb') as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.rstrip(b'\r\n')
            if eng.match_line(raw):
                # Affiche le résultat : chemin:ligne:contenu
                try:
                    text = raw.decode('utf-8')
                except UnicodeDecodeError:
                    text = raw.decode('latin-1', errors='replace')
                print(f"{path}:{i}:{text}")
                count += 1
    return count


def main(argv):
    """
    Point d’entrée du programme en ligne de commande.
    Utilisation :
        python3 egrep_clone.py MOTIF FICHIER
    """
    if len(argv) != 3:
        print("Usage : python3 egrep_clone.py MOTIF FICHIER", file=sys.stderr)
        return 2

    pattern, path = argv[1], argv[2]
    try:
        grep_file(pattern, path)
    except Exception as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))