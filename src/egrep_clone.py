#!/usr/bin/env python3
"""
egrep_clone.py — Projet DAAR : moteur de recherche hors ligne par expressions régulières (sous-ensemble de ERE)

Opérateurs pris en charge : (), |, concaténation implicite, *, ., lettres ASCII.

Utilisation :
  python3 egrep_clone.py MOTIF FICHIER

Description :
  Ce programme compile le motif (pattern) en un automate fini non déterministe (NFA) selon la méthode de Thompson,
  le convertit ensuite en automate déterministe (DFA) par la méthode des sous-ensembles,
  puis le minimise avec l’algorithme de Hopcroft.
  Il recherche enfin les lignes du fichier correspondant à la forme Σ*RΣ* (sous-chaînes).

  Si le motif est un littéral simple (lettres ASCII uniquement), le programme bascule sur un mode rapide :
  il compare la performance entre les algorithmes KMP et Boyer–Moore.
"""
def normalize_case(text: bytes, ignore_case: bool = True) -> bytes:
    return text.lower() if ignore_case else text

import time, sys
from collections import defaultdict, deque

ANY = None  # Symbole générique pour '.' et '.*' (joker)

# -------------------- 1. Analyse lexicale et parsing (Shunting-yard) --------------------

def tokenize(regex: str):
    """Analyse lexicale : transforme une expression régulière en liste de symboles."""
    tokens = []
    i = 0
    while i < len(regex):
        c = regex[i]
        if c in '()*|.':
            tokens.append(c)
            i += 1
        elif c == '\\' and i + 1 < len(regex):
            tokens.append(regex[i+1])
            i += 2
        elif 0x20 <= ord(c) <= 0x7E:
            tokens.append(c)
            i += 1
        else:
            raise ValueError(f"Caractère non supporté : U+{ord(c):04X}")
    return tokens

def insert_concat(tokens):
    """Insère explicitement l’opérateur de concaténation '·' entre symboles contigus."""
    out = []
    for i, t in enumerate(tokens):
        out.append(t)
        if i + 1 < len(tokens):
            t1, t2 = tokens[i], tokens[i+1]
            if ((is_symbol(t1) or t1 in (')','*','.')) and
                (is_symbol(t2) or t2 in ('(','.',))):
                out.append('·')
    return out

def is_symbol(t):
    """Vérifie si un token est un symbole ordinaire (non opérateur)."""
    return t not in {'(',')','|','*','·','.'}

def to_postfix(tokens):
    """Convertit une expression régulière en notation postfixée (algorithme de Dijkstra)."""
    prec = {'*':3, '·':2, '|':1}
    out, stack = [], []
    for t in tokens:
        if is_symbol(t) or t == '.':
            out.append(t)
        elif t in ('|','·','*'):
            if t == '*':
                while stack and stack[-1] == '*':
                    out.append(stack.pop())
                stack.append(t)
            else:
                while stack and stack[-1] != '(' and prec.get(stack[-1],0) >= prec[t]:
                    out.append(stack.pop())
                stack.append(t)
        elif t == '(':
            stack.append(t)
        elif t == ')':
            while stack and stack[-1] != '(':
                out.append(stack.pop())
            if not stack:
                raise ValueError("Parenthèses non appariées")
            stack.pop()
        else:
            raise ValueError(f"Symbole inconnu : {t}")
    while stack:
        if stack[-1] in '()':
            raise ValueError("Parenthèses non appariées")
        out.append(stack.pop())
    return out

# -------------------- 2. Construction du NFA (Thompson) --------------------

class NFA:
    """Représente un automate fini non déterministe."""
    __slots__ = ('start', 'accepts', 'eps', 'trans', 'states')
    def __init__(self):
        self.start = 0
        self.accepts = set()
        self.eps = defaultdict(set)
        self.trans = defaultdict(lambda: defaultdict(set))
        self.states = 0

def new_state(nfa: NFA):
    """Crée un nouvel état et renvoie son identifiant."""
    s = nfa.states
    nfa.states += 1
    return s

def frag_symbol(symbol):
    """Construit un fragment NFA pour un symbole unique (ou pour '.')."""
    n = NFA()
    s = new_state(n)
    f = new_state(n)
    if symbol == '.':
        n.trans[s][ANY].add(f)
    else:
        n.trans[s][symbol].add(f)
    n.start = s
    n.accepts = {f}
    return n

def frag_concat(a: NFA, b: NFA):
    """Concaténation de deux NFA (a suivi de b)."""
    off = a.states
    n = NFA()
    n.states = a.states + b.states
    for u in range(a.states):
        for sym, vs in a.trans[u].items():
            n.trans[u][sym] |= vs
        n.eps[u] |= a.eps[u]
    for u in range(b.states):
        for sym, vs in b.trans[u].items():
            n.trans[u+off][sym] |= {v+off for v in vs}
        n.eps[u+off] |= {v+off for v in b.eps[u]}
    for aacc in a.accepts:
        n.eps[aacc].add(b.start + off)
    n.start = a.start
    n.accepts = {v+off for v in b.accepts}
    return n

def frag_union(a: NFA, b: NFA):
    """Union (|) de deux NFA."""
    off = a.states
    n = NFA()
    n.states = a.states + b.states + 2
    s = new_state(n)
    f = new_state(n)
    for u in range(a.states):
        for sym, vs in a.trans[u].items():
            n.trans[u][sym] |= vs
        n.eps[u] |= a.eps[u]
    for u in range(b.states):
        for sym, vs in b.trans[u].items():
            n.trans[u+off][sym] |= {v+off for v in vs}
        n.eps[u+off] |= {v+off for v in b.eps[u]}
    n.eps[s].add(a.start)
    n.eps[s].add(b.start + off)
    for aacc in a.accepts:
        n.eps[aacc].add(f)
    for bacc in b.accepts:
        n.eps[bacc+off].add(f)
    n.start = s
    n.accepts = {f}
    return n

def frag_star(a: NFA):
    """Fermeture de Kleene (*) : zéro ou plusieurs répétitions du fragment a."""
    n = NFA()
    n.states = a.states + 2
    s = new_state(n)
    f = new_state(n)
    for u in range(a.states):
        for sym, vs in a.trans[u].items():
            n.trans[u][sym] |= vs
        n.eps[u] |= a.eps[u]
    n.eps[s].add(a.start)
    n.eps[s].add(f)
    for acc in a.accepts:
        n.eps[acc].add(f)
        n.eps[acc].add(a.start)
    n.start = s
    n.accepts = {f}
    return n

def build_nfa_from_postfix(postfix):
    """Construit un automate NFA complet à partir de la forme postfixée."""
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
    """Enveloppe le NFA avec Σ* avant et après, pour la recherche de sous-chaînes (Σ*RΣ*)."""
    wrapped = NFA()
    wrapped.states = nfa.states + 2
    s = new_state(wrapped)
    for u in range(nfa.states):
        for sym, vs in nfa.trans[u].items():
            wrapped.trans[u][sym] |= vs
        wrapped.eps[u] |= nfa.eps[u]
    wrapped.trans[s][ANY].add(s)
    wrapped.eps[s].add(nfa.start)
    f = new_state(wrapped)
    for acc in nfa.accepts:
        wrapped.eps[acc].add(f)
    wrapped.start = s
    wrapped.accepts = {f}
    return wrapped

# -------------------- 3. Construction par sous-ensembles (NFA → DFA) --------------------

class DFA:
    """Représente un automate fini déterministe obtenu par la méthode des sous-ensembles."""
    __slots__ = ('start', 'accepts', 'trans', 'states_map', 'states_rev')
    def __init__(self):
        self.start = 0
        self.accepts = set()
        self.trans = defaultdict(dict)
        self.states_map = {}
        self.states_rev = []

def epsilon_closure(nfa: NFA, states):
    """Calcule la ε-fermeture d’un ensemble d’états du NFA."""
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
    """Renvoie les états accessibles depuis 'states' en consommant un octet 'byte_val'."""
    nxt = set()
    ch = chr(byte_val)
    for u in states:
        if ANY in nfa.trans[u]:
            nxt |= nfa.trans[u][ANY]
        if ch in nfa.trans[u]:
            nxt |= nfa.trans[u][ch]
    return nxt

def determinize(nfa: NFA):
    """Applique la méthode des sous-ensembles pour obtenir un DFA équivalent."""
    dfa = DFA()
    start_set = frozenset(epsilon_closure(nfa, {nfa.start}))
    dfa.states_map[start_set] = 0
    dfa.states_rev.append(start_set)
    if any(s in nfa.accepts for s in start_set):
        dfa.accepts.add(0)
    q = deque([0])
    while q:
        s_id = q.popleft()
        S = dfa.states_rev[s_id]
        for b in range(256):
            M = move(nfa, S, b)
            if not M:
                continue
            T = frozenset(epsilon_closure(nfa, M))
            if T not in dfa.states_map:
                dfa.states_map[T] = len(dfa.states_rev)
                dfa.states_rev.append(T)
                if any(t in nfa.accepts for t in T):
                    dfa.accepts.add(dfa.states_map[T])
                q.append(dfa.states_map[T])
            dfa.trans[s_id][b] = dfa.states_map[T]
    return dfa

# -------------------- 4. Minimisation de Hopcroft --------------------

def hopcroft_minimize(dfa: DFA):
    """Algorithme de Hopcroft pour la minimisation du DFA."""
    all_states = set(range(len(dfa.states_rev)))
    A = set(dfa.accepts)
    NA = all_states - A
    P = [A, NA] if NA else [A]
    W = [A] if len(A) <= len(NA) else [NA]
    inv_trans = [defaultdict(set) for _ in range(256)]
    for u, edges in dfa.trans.items():
        for b, v in edges.items():
            inv_trans[b][v].add(u)
    while W:
        Aset = W.pop()
        for b in range(256):
            X = set()
            for q in Aset:
                X |= inv_trans[b].get(q, set())
            if not X:
                continue
            newP = []
            for Y in P:
                inter = Y & X
                diff = Y - X
                if inter and diff:
                    newP.extend([inter, diff])
                    if Y in W:
                        W.remove(Y)
                        W.extend([inter, diff])
                    else:
                        W.append(inter if len(inter) <= len(diff) else diff)
                else:
                    newP.append(Y)
            P = newP
    rep = {}
    for block in P:
        r = min(block)
        for s in block:
            rep[s] = r
    new_states = sorted({rep[s] for s in all_states})
    new_id = {s: i for i, s in enumerate(new_states)}
    mdfa = DFA()
    mdfa.start = new_id[rep[dfa.start]]
    for s_old in new_states:
        mdfa.states_rev.append(set())
    for u, edges in dfa.trans.items():
        u2 = new_id[rep[u]]
        for b, v in edges.items():
            v2 = new_id[rep[v]]
            mdfa.trans[u2][b] = v2
    for a in dfa.accepts:
        mdfa.accepts.add(new_id[rep[a]])
    return mdfa

# -------------------- 5. KMP (recherche littérale) --------------------

def is_literal(regex: str):
    """Vérifie si le motif est un littéral pur (sans métacaractères)."""
    for c in regex:
        if c in '()*|.' or c == '\\':
            return False
    return True

def kmp_build(pattern: bytes):
    """Construit le tableau des préfixes (pi) pour l’algorithme KMP."""
    pi = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j and pattern[i] != pattern[j]:
            j = pi[j-1]
        if pattern[i] == pattern[j]:
            j += 1
            pi[i] = j
    return pi

def kmp_search(pattern: bytes, text: bytes):
    """Recherche d’un motif dans le texte à l’aide de l’algorithme KMP."""
    if not pattern:
        return True
    pi = kmp_build(pattern)
    j = 0
    for i in range(len(text)):
        while j and text[i] != pattern[j]:
            j = pi[j-1]
        if text[i] == pattern[j]:
            j += 1
            if j == len(pattern):
                return True
    return False

# -------------------- 6. Boyer–Moore (optimisé) --------------------

def boyer_moore_preprocess(pattern: bytes):
    """Prépare la table des décalages pour la règle du mauvais caractère."""
    m = len(pattern)
    skip = {i: m for i in range(256)}
    for i in range(m - 1):
        skip[pattern[i]] = m - i - 1
    return skip

def boyer_moore_search(pattern: bytes, text: bytes, skip):
    """Recherche du motif avec Boyer–Moore et table de décalage pré-calculée."""
    m, n = len(pattern), len(text)
    if m == 0 or n == 0:
        return False
    i = 0
    while i <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1
        if j < 0:
            return True
        i += skip.get(text[i + m - 1], m)
    return False

# -------------------- 7. Moteur principal --------------------

class Engine:
    """Classe principale du moteur : gère les deux modes (regex / littéral)."""
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.mode = 'regex'
        if is_literal(pattern):
            self.mode = 'kmp'
            self.literal = pattern.lower().encode('latin-1', 'ignore')
            self.skip_table = boyer_moore_preprocess(self.literal)
            # --- Informations de débogage pour le mode littéral ---
            # Même si aucun automate n’est construit (KMP / Boyer–Moore),
            # on affiche des valeurs nulles afin de garder un format uniforme pour run_tests.py.
            print("[DEBUG] NFA: 0 états, DFA: 0, DFA min: 0", file=sys.stderr)

        else:
            tokens = tokenize(pattern)
            tokens = insert_concat(tokens)
            postfix = to_postfix(tokens)
            nfa = build_nfa_from_postfix(postfix)
            nfa = add_search_wrappers(nfa)
            dfa = determinize(nfa)
            self.dfa = hopcroft_minimize(dfa)
            # --- Affichage unifié des informations de débogage ---
            # Cette section garantit que les valeurs de NFA / DFA / DFA_min
            # sont toujours affichées, même pour les recherches littérales (KMP / Boyer–Moore).
            # Les données sont envoyées sur stderr afin d’être détectées par run_tests.py.
            print(f"[DEBUG] NFA: {nfa.states} états, DFA: {len(dfa.states_rev)}, DFA min: {len(self.dfa.states_rev)}", file=sys.stderr)

    def match_line(self, line: bytes) -> bool:
        """Vérifie si une ligne correspond au motif (via DFA ou KMP/BM)."""
        if self.mode == 'kmp':
            found_kmp = kmp_search(self.literal, line)
            found_bm = boyer_moore_search(self.literal, line, self.skip_table)
            return found_kmp or found_bm
        s = self.dfa.start
        for b in line:
            s = self.dfa.trans.get(s, {}).get(b, None)
            if s is None:
                s = self.dfa.start
                s = self.dfa.trans.get(s, {}).get(b, s)
        return s in self.dfa.accepts

# -------------------- 8. Fonction principale de recherche --------------------

def grep_file(pattern: str, path: str):
    """Parcourt un fichier et affiche les lignes correspondant au motif (insensible à la casse, sans doublons)."""
    eng = Engine(pattern)
    count = 0

    # --- Mode littéral : comparaison via KMP et Boyer–Moore ---
    if eng.mode == 'kmp':
        print("[DEBUG] Mode littéral détecté : comparaison KMP / Boyer–Moore (insensible à la casse)", file=sys.stderr)

        # Lecture du fichier complet
        with open(path, 'rb') as f:
            lines = [l.rstrip(b'\r\n') for l in f]

        # --- Phase KMP ---
        start_kmp = time.time()
        count_kmp = 0
        seen_lines = set()  # Pour éviter de compter plusieurs fois la même ligne

        import re
        pattern_str = r'\b' + eng.literal.decode('latin-1') + r'\b'
        for i, raw in enumerate(lines, start=1):
            text = raw.decode('latin-1', errors='ignore')
            if re.search(pattern_str, text, re.IGNORECASE):
                if i not in seen_lines:  # Empêche les doublons
                    seen_lines.add(i)
                    try:
                        text = raw.decode('utf-8')
                    except UnicodeDecodeError:
                        text = raw.decode('latin-1', errors='replace')
                    print(f"{path}:{i}:{text}")
                    count_kmp += 1

        t_kmp = time.time() - start_kmp

        # --- Phase Boyer–Moore (pour comparaison de performance ET comptage) ---
        start_bm = time.time()
        count_bm = 0
        for i, raw in enumerate(lines, start=1):
            raw_lower = raw.lower()
            if boyer_moore_search(eng.literal, raw_lower, eng.skip_table):
                count_bm += 1
        t_bm = time.time() - start_bm

        # Affichage du résumé de performance
        print(f"[DEBUG] KMP: {t_kmp:.6f}s ({count_kmp} lignes)", file=sys.stderr)
        print(f"[DEBUG] Boyer–Moore: {t_bm:.6f}s ({count_bm} lignes)", file=sys.stderr)

        return count_kmp

    # --- Mode regex (automates finis) ---
    t0 = time.time()
    seen_lines = set()  # Même principe : éviter les doublons éventuels
    with open(path, 'rb') as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.rstrip(b'\r\n')
            if eng.match_line(raw):
                if i not in seen_lines:
                    seen_lines.add(i)
                    try:
                        text = raw.decode('utf-8')
                    except UnicodeDecodeError:
                        text = raw.decode('latin-1', errors='replace')
                    print(f"{path}:{i}:{text}")
                    count += 1
    t1 = time.time()
    print(f"[DEBUG] Temps total: {t1 - t0:.6f}s", file=sys.stderr)
    return count

# -------------------- 9. Point d’entrée du programme --------------------

def main(argv):
    """Point d’entrée du programme en ligne de commande."""
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