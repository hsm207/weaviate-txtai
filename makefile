setup:
	poetry config virtualenvs.path $$(pwd)/.venv && \
		poetry install

install: setup
	pip install --editable .

build:
	poetry build

publish: build
	poetry publish

clean:
	-rm -r dist
	-pip uninstall -y weaviate-txtai
	-rm -r .venv