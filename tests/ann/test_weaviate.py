import subprocess
import time

import pytest
import weaviate
from txtai.app import Application
from txtai.embeddings import Embeddings
from txtai.pipeline.nop import Nop
from txtai.workflow import Task, Workflow

import weaviate_txtai.ann.weaviate as ann
from weaviate_txtai.client import Weaviate

WEAVIATE_DB_URL = "http://localhost:8080"


@pytest.fixture
def weaviate_db():
    subprocess.run(
        ["docker-compose", "-p", "weaviate_test_server", "up", "-d"],
        check=True,
        capture_output=True,
    )

    # wait for deployment to be ready
    time.sleep(1)

    yield

    subprocess.run(["docker-compose", "-p", "weaviate_test_server", "down"], check=True)


@pytest.fixture
def weaviate_client():
    yield weaviate.Client(WEAVIATE_DB_URL)


def test_default_schema(weaviate_db, weaviate_client):
    default_schema = {
        "class": "Document",
        "properties": [{"name": "docid", "dataType": ["int"]}],
        "vectorIndexConfig": {"distance": "cosine"},
    }

    config = {"weaviate": {"url": WEAVIATE_DB_URL}}

    ann_weaviate = ann.Weaviate(config)
    assert weaviate_client.schema.contains(default_schema)


def test_custom_schema(weaviate_db, weaviate_client):
    custom_schema = {
        "class": "Post",
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "docid", "dataType": ["int"]},
        ],
        "vectorIndexConfig": {"distance": "dot"},
    }

    config = {"weaviate": {"url": WEAVIATE_DB_URL, "schema": custom_schema}}

    ann_weaviate = ann.Weaviate(config)
    assert weaviate_client.schema.contains(custom_schema)




def test_count(weaviate_db):
    embeddings = Embeddings(
        {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "backend": "weaviate_txtai.ann.weaviate.Weaviate",
        }
    )

    docs = [(0, "Lorem ipsum", None), (1, "dolor sit amet", None)]
    embeddings.index(docs)

    assert embeddings.count() == len(docs)


def test_invalid_schema(weaviate_db):
    invalid_schema = {
        "class": "Article",
        "properties": [{"name": "content", "dataType": ["text"]}],
    }

    config = {"weaviate": {"url": WEAVIATE_DB_URL, "schema": invalid_schema}}

    with pytest.raises(weaviate.exceptions.SchemaValidationException):
        ann.Weaviate(config)


def test_index(weaviate_db, weaviate_client):

    embeddings = Embeddings(
        {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "backend": "weaviate_txtai.ann.weaviate.Weaviate",
        }
    )

    embeddings.index([(0, "Lorem ipsum", None), (1, "dolor sit amet", None)])
    total_docs = 2

    assert embeddings.ann.config["offset"] == total_docs

    results = weaviate_client.data_object.get(class_name="Document", with_vector=True)

    assert results["totalResults"] == total_docs

    objects = results["objects"]
    assert all([obj["vector"] for obj in objects])


def test_search(weaviate_db, weaviate_client):

    embeddings = Embeddings(
        {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "backend": "weaviate_txtai.ann.weaviate.Weaviate",
        }
    )

    embeddings.index(
        [
            ("foo", "the quick brown fox", None),
            ("bar", "jumps over the lazy dog", None),
            ("baz", "Stock futures fall after post-Powell rally", None),
        ]
    )

    # vixen is closer to the first sentence
    result = embeddings.search("vixen", 3)
    assert result[0][0] == "foo"

    # puppy is closer to the second sentence
    result = embeddings.search("puppy", 3)
    assert result[0][0] == "bar"

    # financial markets are closer to the third sentence
    result = embeddings.search("financial markets", 3)
    assert result[0][0] == "baz"


def test_save(weaviate_db):

    embeddings = Embeddings(
        {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "backend": "weaviate_txtai.ann.weaviate.Weaviate",
        }
    )

    embeddings.index([(0, "Lorem ipsum", None)])

    with pytest.raises(NotImplementedError, match=r"not yet supported"):
        embeddings.save("test")


def test_load(weaviate_db):
    embeddings = Embeddings(
        {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "backend": "weaviate_txtai.ann.weaviate.Weaviate",
        }
    )

    with pytest.raises(NotImplementedError, match=r"not yet supported"):
        embeddings = Embeddings()
        embeddings.load("test")


def test_delete(weaviate_db, weaviate_client):
    embeddings = Embeddings(
        {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "backend": "weaviate_txtai.ann.weaviate.Weaviate",
        }
    )

    embeddings.index([(0, "Lorem ipsum", None)])
    objects = weaviate_client.data_object.get(class_name="Document")["objects"]

    assert len(objects) == 1

    embeddings.delete([0])
    objects = weaviate_client.data_object.get(class_name="Document")["objects"]

    assert len(objects) == 0
