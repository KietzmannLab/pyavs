# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build (warnings-as-errors) and serve the site locally at http://localhost:$(PORT).
# Blocks in the foreground -- stop with Ctrl-C. Usage: make serve [PORT=8765]
PORT ?= 8765
.PHONY: serve
serve:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" -W $(SPHINXOPTS) $(O)
	@echo "Serving pyAVS docs at http://localhost:$(PORT)/index.html  (Ctrl-C to stop)"
	@cd "$(BUILDDIR)/html" && python3 -m http.server $(PORT)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx-build using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)