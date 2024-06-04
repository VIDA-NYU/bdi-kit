SRC := ./bdikit/

all: lint test

PHONY: format test lint

lint:
	black --check ${SRC}

test:
	python3 -m pytest

format:
	black ${SRC}
