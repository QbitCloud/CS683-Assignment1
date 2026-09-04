TYPST ?= $(HOME)/.local/bin/typst

.PHONY: all watch clean
all: report.pdf

report.pdf: report.typ template.typ $(wildcard figures/*)
	$(TYPST) compile report.typ

watch:
	$(TYPST) watch report.typ

clean:
	rm -f report.pdf page-*.png
