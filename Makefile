# Makefile minimal pour le prototype Python
PY=python3
APP=src/egrep_clone.py
ZIPNAME=daar-projet-offline-NOM1-NOM2.zip

.PHONY: test zip

# ==== Test général : expressions régulières ====
test:
	@echo "\033[1;33m===============================================\033[0m"
	@echo "\033[1;33m== Démo locale : mode AUTOMATE (NFA→DFA→DFAmin) ==\033[0m"
	@echo "\033[1;33m===============================================\033[0m"
	$(PY) $(APP) 'ab*a' tests/demo.txt || true
	$(PY) $(APP) '(ab)*c' tests/demo.txt || true
	$(PY) $(APP) '(a|b)*a' tests/demo.txt || true
	$(PY) $(APP) '....a' tests/demo.txt || true
	$(PY) $(APP) 'S(a|g|r)*on' tests/demo.txt || true
	@echo ""

# ==== Test KMP : motifs littéraux sans opérateurs ====
test-kmp:
	@echo "\033[1;32m===============================================\033[0m"
	@echo "\033[1;32m== Démo locale : mode KMP (recherche littérale) ==\033[0m"
	@echo "\033[1;32m===============================================\033[0m"
	$(PY) $(APP) 'bonjour' tests/demo.txt || true
	$(PY) $(APP) 'Alice' tests/demo.txt || true
	$(PY) $(APP) 'Paris' tests/demo.txt || true
	@echo ""

# ==== Test global (Automate + KMP) ====
test-all: test test-kmp
	@echo "==============================================="
	@echo "== Tous les tests terminés (Automate + KMP) =="
	@echo "==============================================="

# ==== Archive finale ====
zip:
	@echo "== Archive du rendu =="
	@rm -f $(ZIPNAME)
	zip -r $(ZIPNAME) src/ tests/ report/ README.md Makefile
	@echo "Produit: $(ZIPNAME)"