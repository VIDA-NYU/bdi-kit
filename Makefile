SRC := ./bdikit/ ./tests/ ./scripts/

all: lint test

PHONY: format test lint

lint:
	black --check ${SRC}

test:
	pytest ./tests/

format:
	black ${SRC}
