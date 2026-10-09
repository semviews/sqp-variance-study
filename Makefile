PY ?= SemBench/.venvs/sembench/bin/python

.PHONY: rebuild check test clean

# Regenerate every processed file, and every table, figure, and macro under outputs/, from the raw logs.
rebuild:
	PY=$(PY) scripts/rebuild.sh

# Compare the regenerated tables and macros with the submitted copy in reference/.
check:
	python3 scripts/compare_reference.py

test:
	$(PY) -m pytest -q -p no:cacheprovider tests

clean:
	rm -rf outputs
