# pdfcancel development helpers
#
# After editing source code, run `make install` to update the installed package.
# Homebrew Python's venv doesn't support editable installs reliably,
# so we use a standard install and reinstall after changes.

VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python3.12
PDFCANCEL := $(VENV)/bin/pdfcancel

.PHONY: install test clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install/reinstall pdfcancel into the venv
	$(PIP) install --quiet .

dev-install: ## Install with dev dependencies
	$(PIP) install --quiet ".[dev]"

test: ## Run a quick smoke test on the ref/ PDF
	$(PDFCANCEL) ref/ -o /tmp/pdfcancel_test --force --verbose

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
