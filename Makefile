# Convenience targets. Run `make help` to list them.
.PHONY: help install test lint traces demo run clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package + dev dependencies
	pip install -e ".[dev]"

test:  ## Run the test suite
	pytest -q

lint:  ## Lint with ruff
	ruff check src tests

traces:  ## Capture real workload traces into data/traces/ (needs a C compiler)
	bash scripts/capture_traces.sh

demo:  ## Run the full benchmark and regenerate all charts in results/
	cachesim demo

run:  ## Quick benchmark on a real trace (override: make run TRACE=... CAP=...)
	cachesim run --trace $(or $(TRACE),data/traces/bst.txt) --capacity $(or $(CAP),64)

clean:  ## Remove caches and generated outputs
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info
