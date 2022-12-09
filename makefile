setup:
	poetry config virtualenvs.in-project true && \
		poetry install && \
		docker-compose pull

install: setup
	pip install --editable .

build:
	poetry build

publish: build
	poetry publish

test:
	poetry run pytest

coverage:
	poetry run coverage run -m pytest && \
		poetry run coverage report -m
format:
	poetry run black .
	
clean:
	-rm -r dist
	-pip uninstall -y weaviate-txtai
	-rm -r .venv