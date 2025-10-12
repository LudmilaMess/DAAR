# ==== Makefile minimal pour le prototype Python ====
PY=python3
APP=src/egrep_clone.py
ZIPNAME=daar-egrep-LiuYANG-LudmilaMessaoudi.zip

.PHONY: test test-kmp test-all clean zip

.DEFAULT_GOAL := test-all

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
	@echo "==============================================="
	@echo "== Démo locale : mode KMP (recherche littérale) =="
	@echo "==============================================="
	$(PY) $(APP) 'alpha'  tests/demo.txt || true
	$(PY) $(APP) 'abba'   tests/demo.txt || true
	$(PY) $(APP) 'Sargon' tests/demo.txt || true
	$(PY) $(APP) 'Saon'   tests/demo.txt || true

# ==== Test global (Automate + KMP) ====
test-all: test test-kmp
	@echo "==============================================="
	@echo "== Tous les tests terminés (Automate + KMP) =="
	@echo "==============================================="

# ==== Nettoyage ====
clean:
	@echo "== Nettoyage des fichiers temporaires =="
	@rm -rf __pycache__ tests/__pycache__ tests/outputs/*.png results.csv

# ==== Archive finale ====
zip:
	@echo "== Archive du rendu =="
	@rm -f $(ZIPNAME)
	zip -r $(ZIPNAME) src/ tests/ README.md Makefile -x "tests/outputs/*"
	@echo "Produit: $(ZIPNAME)"