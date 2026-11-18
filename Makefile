.PHONY: setup start test test-unit test-integration benchmark doctor build-web build-llama clean

setup:
	./scripts/setup.sh

start:
	./scripts/start.sh

test:
	.venv/bin/pytest -q

test-unit:
	.venv/bin/pytest -q -m "not integration"

test-integration:
	.venv/bin/pytest -q -m integration

benchmark:
	.venv/bin/python scripts/benchmark.py

doctor:
	.venv/bin/python -m app doctor

build-web:
	cd frontend && npm install --no-audit --no-fund && npm run build

build-llama:
	./scripts/build_llama.sh

clean:
	rm -rf .cache .pytest_cache app/static/dashboard/assets frontend/dist
