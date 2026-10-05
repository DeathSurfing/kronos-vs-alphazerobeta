# Paper build. Requires tectonic (or any XeLaTeX-compatible engine).
#   make paper     -> kronos_vs_alphazerobeta.pdf
#   make clean
TEX := kronos_vs_alphazerobeta
TECTONIC ?= tectonic

.PHONY: paper clean tables figures all

paper: $(TEX).pdf

$(TEX).pdf: $(TEX).tex $(TEX).bib $(wildcard tables/*.tex) $(wildcard ../results/figures/*.pdf)
	$(TECTONIC) $(TEX).tex
	@echo "built $@"

tables:
	python3 ../code/statistics/generate_tables.py
	python3 ../code/statistics/to_latex.py

figures:
	python3 ../code/statistics/generate_figures.py

all: tables figures paper

clean:
	rm -f $(TEX).pdf $(TEX).aux $(TEX).log $(TEX).out $(TEX).bbl $(TEX).blg
