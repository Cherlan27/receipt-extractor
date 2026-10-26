.PHONY: check test lint specs

# The project check: must pass at the end of apply and fixes, and runs in CI.
check: test lint specs

test:
	poetry run pytest -q

lint:
	poetry run pre-commit run --all-files

specs:
	openspec validate --all --strict
