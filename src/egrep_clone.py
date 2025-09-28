#!/usr/bin/env python3
"""
egrep_clone.py — DAAR Projet 1: offline regex search (subset of ERE)
Supported operators: (), |, concatenation (implicit), *, ., ASCII letters.

Usage:
  python egrep_clone.py PATTERN FILE

This prototype compiles PATTERN to an NFA (Thompson), converts to DFA (subset),
minimizes via Hopcroft, and searches each line for a substring match (Σ* R Σ*).
If the pattern is a plain literal (ASCII letters only), it uses KMP.
"""

import sys
from collections import defaultdict, deque

ANY = None  # wildcard symbol for dot '.' and '.*' loops

# -------------------- Tokenization & Shunting-yard --------------------

def tokenize(regex: str):
    tokens = []
    i = 0
    while i < len(regex):
        c = regex[i]
        if c in '()*|.':
            tokens.append(c)
            i += 1
        elif c == '\\' and i + 1 < len(regex):
            # escape next char as literal
            tokens.append(regex[i+1])
            i += 2
        elif 0x20 <= ord(c) <= 0x7E:  # printable ASCII as literal symbol
            tokens.append(c)
            i += 1
        else:
            raise ValueError(f"Unsupported character in pattern: U+{ord(c):04X}")
    return tokens

def insert_concat(tokens):
    out = []
    for i, t in enumerate(tokens):
        out.append(t)
        if i + 1 < len(tokens):
            t1, t2 = tokens[i], tokens[i+1]
            # If t1 is symbol, ')', or '*', and t2 is symbol, '(' or '.', insert concatenation '·'
            if ((is_symbol(t1) or t1 in (')','*','.')) and
                (is_symbol(t2) or t2 in ('(','.',))):
                out.append('·')  # explicit concatenation operator
    return out

def is_symbol(t):
    return t not in {'(',')','|','*','·','.'}

def to_postfix(tokens):
    prec = {'*':3, '·':2, '|':1}
    out = []
    stack = []
    for t in tokens:
        if is_symbol(t) or t == '.':
            out.append(t)
        elif t in ('|','·','*'):
            if t == '*':
                # unary postfix — pop while top has higher precedence (i.e., '*')
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
                raise ValueError("Mismatched parentheses")
            stack.pop()
        else:
            raise ValueError(f"Unknown token: {t}")
    while stack:
        if stack[-1] in '()':
            raise ValueError("Mismatched parentheses")
        out.append(stack.pop())
    return out

# -------------------- Thompson NFA --------------------

class NFA:
    __slots__ = ('start', 'accepts', 'eps', 'trans', 'states')
    def __init__(self):
        self.start = 0
        self.accepts = set()
        self.eps = defaultdict(set)       # epsilon transitions: state -> {state}
        self.trans = defaultdict(lambda: defaultdict(set))  # state -> {symbol -> {state}}
        self.states = 0

def new_state(nfa: NFA):
    s = nfa.states
    nfa.states += 1
    return s

def frag_symbol(symbol):
    n = NFA()
    s = new_state(n); f = new_state(n)
    if symbol == '.':
        n.trans[s][ANY].add(f)  # wildcard
    else:
        n.trans[s][symbol].add(f)
    n.start = s
    n.accepts = {f}
    return n

def frag_concat(a: NFA, b: NFA):
    # offset b by a.states
    off = a.states
    n = NFA()
    n.states = a.states + b.states
    # copy trans/eps
    for u in range(a.states):
        for sym, vs in a.trans[u].items():
            n.trans[u][sym] |= {v for v in vs}
        n.eps[u] |= set(a.eps[u])
    for u in range(b.states):
        for sym, vs in b.trans[u].items():
            n.trans[u+off][sym] |= {v+off for v in vs}
        n.eps[u+off] |= {v+off for v in b.eps[u]}
    # connect a.accepts -> b.start via epsilon
    for aacc in a.accepts:
        n.eps[aacc].add(b.start + off)
    n.start = a.start
    n.accepts = {v+off for v in b.accepts}
    return n

def frag_union(a: NFA, b: NFA):
    off = a.states
    n = NFA()
    n.states = a.states + b.states + 2
    s = new_state(n)  # new start
    f = new_state(n)  # new accept
    # copy a
    for u in range(a.states):
        for sym, vs in a.trans[u].items():
            n.trans[u][sym] |= set(vs)
        n.eps[u] |= set(a.eps[u])
    # copy b (offset)
    for u in range(b.states):
        for sym, vs in b.trans[u].items():
            n.trans[u+off][sym] |= {v+off for v in vs}
        n.eps[u+off] |= {v+off for v in b.eps[u]}
    # connect
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
    n = NFA()
    n.states = a.states + 2
    s = new_state(n)
    f = new_state(n)
    # copy a
    for u in range(a.states):
        for sym, vs in a.trans[u].items():
            n.trans[u][sym] |= set(vs)
        n.eps[u] |= set(a.eps[u])
    # connect
    n.eps[s].add(a.start)
    n.eps[s].add(f)
    for acc in a.accepts:
        n.eps[acc].add(f)
        n.eps[acc].add(a.start)
    n.start = s
    n.accepts = {f}
    return n

def build_nfa_from_postfix(postfix):
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
            raise ValueError(f"Unknown postfix token {t}")
    if len(stack) != 1:
        raise ValueError("Invalid postfix expression")
    return stack[0]

def add_search_wrappers(nfa: NFA):
    """Wrap NFA as Σ*  (nfa)  Σ*, so we can match substrings by full-line acceptance."""
    # Prefix Σ*
    wrapped = NFA()
    wrapped.states = nfa.states + 2
    s = new_state(wrapped)  # new start
    for u in range(nfa.states):
        for sym, vs in nfa.trans[u].items():
            wrapped.trans[u][sym] |= set(vs)
        wrapped.eps[u] |= set(nfa.eps[u])
    # loop on ANY at new start
    wrapped.trans[s][ANY].add(s)
    wrapped.eps[s].add(nfa.start)
    # Suffix Σ*
    f = new_state(wrapped)  # new accept
    for acc in nfa.accepts:
        wrapped.eps[acc].add(f)
    wrapped.trans[f][ANY].add(f)  # absorb anything after match
    wrapped.start = s
    wrapped.accepts = {f}
    return wrapped

# -------------------- Subset Construction (NFA -> DFA) --------------------

class DFA:
    __slots__ = ('start', 'accepts', 'trans', 'states_map', 'states_rev')
    def __init__(self):
        self.start = 0
        self.accepts = set()
        self.trans = defaultdict(dict)  # state -> {byte -> state}
        self.states_map = {}            # frozenset(NFA states) -> dfa_state_id
        self.states_rev = []            # reverse mapping

def epsilon_closure(nfa: NFA, states):
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
    """Compute set of NFA states reachable from 'states' on input byte 'byte_val'."""
    nxt = set()
    ch = chr(byte_val)
    for u in states:
        if ANY in nfa.trans[u]:
            nxt |= nfa.trans[u][ANY]
        if ch in nfa.trans[u]:
            nxt |= nfa.trans[u][ch]
    return nxt

def determinize(nfa: NFA):
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
        # compute transitions for all 256 byte values
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

# -------------------- Hopcroft Minimization --------------------

def hopcroft_minimize(dfa: DFA):
    # Initialize partition P with accepting and non-accepting sets
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

    # Build new DFA
    rep = {}
    for block in P:
        r = min(block)
        for s in block:
            rep[s] = r
    new_states = sorted({rep[s] for s in all_states})
    new_id = {s:i for i,s in enumerate(new_states)}

    mdfa = DFA()
    mdfa.start = new_id[rep[dfa.start]]
    for s_old in new_states:
        mdfa.states_rev.append(set())  # placeholder
    for u, edges in dfa.trans.items():
        u2 = new_id[rep[u]]
        for b, v in edges.items():
            v2 = new_id[rep[v]]
            mdfa.trans[u2][b] = v2
    for a in dfa.accepts:
        mdfa.accepts.add(new_id[rep[a]])
    return mdfa

# -------------------- KMP (literal fast path) --------------------

def is_literal(regex: str):
    for c in regex:
        if c in '()*|.' or c == '\\':
            return False
    return True

def kmp_build(pattern: bytes):
    pi = [0]*len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j and pattern[i] != pattern[j]:
            j = pi[j-1]
        if pattern[i] == pattern[j]:
            j += 1
            pi[i] = j
    return pi

def kmp_search(pattern: bytes, text: bytes):
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

# -------------------- High-level compile & match --------------------

class Engine:
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.mode = 'regex'
        if is_literal(pattern):
            self.mode = 'kmp'
            self.literal = pattern.encode('latin-1', 'ignore')
        else:
            tokens = tokenize(pattern)
            tokens = insert_concat(tokens)
            postfix = to_postfix(tokens)
            nfa = build_nfa_from_postfix(postfix)
            nfa = add_search_wrappers(nfa)
            dfa = determinize(nfa)
            self.dfa = hopcroft_minimize(dfa)

    def match_line(self, line: bytes) -> bool:
        if self.mode == 'kmp':
            return kmp_search(self.literal, line)
        # DFA scan
        s = self.dfa.start
        for b in line:
            s = self.dfa.trans.get(s, {}).get(b, None)
            if s is None:
                # dead transition; not present in DFA graph -> reject (no match yet)
                # but because we compiled Σ* R Σ*, missing edges shouldn't happen often;
                # treat as sink by restarting from start state transition on b
                s = self.dfa.start
                s = self.dfa.trans.get(s, {}).get(b, s)
            # Early exit: if we have an accepting state and our design uses suffix Σ*,
            # we can continue but the final decision will be accepting at end. To be safe
            # we just keep scanning; the final state acceptance is sufficient.
        return s in self.dfa.accepts

def grep_file(pattern: str, path: str):
    eng = Engine(pattern)
    count = 0
    with open(path, 'rb') as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.rstrip(b'\r\n')
            if eng.match_line(raw):
                # print line number and line, loosely mimicking egrep
                try:
                    text = raw.decode('utf-8')
                except UnicodeDecodeError:
                    text = raw.decode('latin-1', errors='replace')
                print(f"{path}:{i}:{text}")
                count += 1
    return count

def main(argv):
    if len(argv) != 3:
        print("Usage: python egrep_clone.py PATTERN FILE", file=sys.stderr)
        return 2
    pattern, path = argv[1], argv[2]
    try:
        grep_file(pattern, path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
