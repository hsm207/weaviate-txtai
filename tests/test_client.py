import subprocess
import time

import pytest
from txtai.app import Application
from txtai.embeddings import Embeddings
from txtai.pipeline.nop import Nop
from txtai.workflow import Task, Workflow

from weaviate_txtai.client import Weaviate
import weaviate

WEAVIATE_DB_URL = "http://localhost:8080"


@pytest.fixture
def weaviate_db():
    subprocess.run(["docker-compose", "up", "-d"], check=True, capture_output=True)

    # wait for deployment to be ready
    time.sleep(1)

    yield

    subprocess.run(["docker-compose", "down"], check=True)


@pytest.fixture
def weaviate_client():
    yield weaviate.Client(WEAVIATE_DB_URL)


@pytest.fixture
def app(weaviate_db):
    workflow = """
    weaviate_txtai.client.Weaviate:

    embeddings:
     path: sentence-transformers/nli-mpnet-base-v2

    nop:

    workflow:
     index:
      tasks:
       - action: [nop, transform]
         unpack: False
       - action: weaviate_txtai.client.Weaviate
         unpack: False
     search:
      tasks:
       - action: transform
       - action: weaviate_txtai.client.Weaviate
         args: [search]
    """
    yield Application(workflow)


def test_default_schema(weaviate_db, weaviate_client):
    workflow = """
    weaviate_txtai.client.Weaviate:

    workflow:
     simple:
      tasks:
       - action: weaviate_txtai.client.Weaviate
    
    """

    app = Application(workflow)
    schema = weaviate_client.schema.get("Document")

    assert schema["class"] == "Document"
    assert any(
        p["name"] == "content" and p["dataType"][0] == "text"
        for p in schema["properties"]
    )
    assert schema["vectorIndexConfig"]["distance"] == "cosine"


def test_custom_schema(weaviate_db, weaviate_client):
    workflow = """
    weaviate_txtai.client.Weaviate:
     custom_schema:
      class: "Post"
      properties:
       - name: "content"
         dataType:
          - text
      vectorIndexConfig:
       distance: "dot"

    workflow:
     simple:
      tasks:
       - action: weaviate_txtai.client.Weaviate
    
    """

    app = Application(workflow)
    schema = weaviate_client.schema.get("Post")
    assert schema["vectorIndexConfig"]["distance"] == "dot"


def test_invalid_schema(weaviate_db):
    workflow = """
    weaviate_txtai.client.Weaviate:
     custom_schema:
      class: "Foo"
      properties:
       - name: "bar"
         dataType:
          - text
      vectorIndexConfig:
       distance: "cosine"

    workflow:
     simple:
      tasks:
       - action: weaviate_txtai.client.Weaviate
    
    """
    with pytest.raises(AssertionError) as e:
        Application(workflow)

    expected_exception_msg = (
        "Custom schema must have a property named 'content' with type 'text'"
    )
    actual_exception_msg = str(e.value)
    assert expected_exception_msg in actual_exception_msg


def test_index_workflow(app, weaviate_client):
    data = ["hello world"]

    list(app.workflow("index", data))
    doc = weaviate_client.query.get("Document", ["content"]).do()["data"]["Get"][
        "Document"
    ]
    assert doc[0]["content"] == data[0]


def test_search_workflow(app):
    workflow = """
    weaviate_txtai.client.Weaviate:

    embeddings:
     path: sentence-transformers/nli-mpnet-base-v2

    nop:

    workflow:
     search:
      tasks:
       - action: transform
       - action: weaviate_txtai.client.Weaviate
         args: [search]
    """

    queries = ["hello world"]

    # when the db is empty
    empty_results = list(app.workflow("search", queries))
    empty_doc = empty_results[0]["data"]["Get"]["Document"]

    # when the db is not empty
    data = ["hi there", "good bye"]
    list(app.workflow("index", data))

    nonempty_results = list(app.workflow("search", queries))
    nonempty_doc = nonempty_results[0]["data"]["Get"]["Document"][0]

    assert not empty_doc
    assert nonempty_doc["content"]
    assert nonempty_doc["_additional"]["certainty"]
