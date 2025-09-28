# Makefile minimal pour le prototype Python
PY=python
APP=src/egrep_clone.py
ZIPNAME=daar-projet-offline-NOM1-NOM2.zip

.PHONY: test zip

test:
	@echo "== Démo locale =="
	$(PY) $(APP) 'ab*a' tests/demo.txt || true
	$(PY) $(APP) '...a' tests/demo.txt || true

zip:
	@echo "== Archive du rendu =="
	@rm -f $(ZIPNAME)
	zip -r $(ZIPNAME) src/ tests/ report/ README.md Makefile
	@echo "Produit: $(ZIPNAME)"
