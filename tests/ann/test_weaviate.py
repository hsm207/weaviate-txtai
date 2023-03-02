import subprocess
import time

import pytest
import weaviate
from txtai.app import Application
from txtai.embeddings import Embeddings
from txtai.pipeline.nop import Nop
from txtai.workflow import Task, Workflow

import weaviate_txtai.ann.weaviate as ann

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


@pytest.fixture
def embeddings(weaviate_db):
    yield Embeddings(
        {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "backend": "weaviate_txtai.ann.weaviate.Weaviate",
        }
    )


def test_default_schema(weaviate_db, weaviate_client):
    default_schema = {
        "class": "Document",
        "properties": [{"name": "docid", "dataType": ["int"]}],
        "vectorIndexConfig": {"distance": "cosine"},
    }

    config = {"weaviate": {"url": WEAVIATE_DB_URL}}

    ann.Weaviate(config)
    assert weaviate_client.schema.contains(default_schema)


def test_custom_schema(weaviate_db, weaviate_client):
    custom_schema = {
        "class": "Post",
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "docid", "dataType": ["int"]},
        ],
    }

    config = {"weaviate": {"url": WEAVIATE_DB_URL, "schema": custom_schema}}

    ann.Weaviate(config)
    assert weaviate_client.schema.contains(custom_schema)


def test_invalid_distance_metric(weaviate_db):
    invalid_schema = {
        "class": "Article",
        "properties": [{"name": "docid", "dataType": ["int"]}],
        "vectorIndexConfig": {"distance": "dot"},
    }

    config = {"weaviate": {"url": WEAVIATE_DB_URL, "schema": invalid_schema}}

    with pytest.raises(weaviate.exceptions.SchemaValidationException):
        ann.Weaviate(config)


def test_overwrite_schema(embeddings):

    docs = [(0, "Lorem ipsum", None), (1, "dolor sit amet", None)]

    embeddings.index(docs)
    embeddings.index(docs)

    assert embeddings.count() == len(docs)


def test_duplicate_schema(weaviate_db, caplog):
    weaviate_config = {"url": WEAVIATE_DB_URL, "overwrite_index": False}

    embeddings = Embeddings(
        {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "backend": "weaviate_txtai.ann.weaviate.Weaviate",
            "weaviate": weaviate_config,
        }
    )

    docs = [(0, "Lorem ipsum", None), (1, "dolor sit amet", None)]

    embeddings.index(docs)
    embeddings.index(docs)

    assert "already exists" in caplog.text


def test_invalid_schema(weaviate_db):
    invalid_schema = {
        "class": "Article",
        "properties": [{"name": "content", "dataType": ["text"]}],
    }

    config = {"weaviate": {"url": WEAVIATE_DB_URL, "schema": invalid_schema}}

    with pytest.raises(weaviate.exceptions.SchemaValidationException):
        ann.Weaviate(config)


def test_count(embeddings):

    docs = [(0, "Lorem ipsum", None), (1, "dolor sit amet", None)]
    embeddings.index(docs)

    assert embeddings.count() == len(docs)


def test_index(embeddings, weaviate_client):

    docs = [(0, "Lorem ipsum", None), (1, "dolor sit amet", None)]
    total_docs = len(docs)

    embeddings.index(docs)

    assert embeddings.ann.config["offset"] == total_docs

    results = weaviate_client.data_object.get(class_name="Document", with_vector=True)

    assert results["totalResults"] == total_docs

    objects = results["objects"]
    assert all([obj["vector"] for obj in objects])


def test_search(embeddings):

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


def test_save_and_load_overwrite(embeddings, caplog, tmp_path):

    savefile = str(tmp_path / "test")

    embeddings.index([(0, "Lorem ipsum", None)])

    embeddings.save(savefile)

    assert "save method has no effect" in caplog.text

    embeddings.load(savefile)

    assert "load method has no effect" in caplog.text


def test_save_and_load_reuse(weaviate_db, weaviate_client, tmp_path):

    savefile = str(tmp_path / "test")

    old_embeddings = Embeddings(
        {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "backend": "weaviate_txtai.ann.weaviate.Weaviate",
            "weaviate": {"url": WEAVIATE_DB_URL, "overwrite_index": False},
        }
    )

    old_embeddings.index([(0, "Lorem ipsum", None)])
    old_shard_name = weaviate_client.schema.get_class_shards("Document")[0]["name"]
    old_embeddings.save(savefile)

    new_embeddings = Embeddings()
    new_embeddings.load(savefile)
    new_shard_name = weaviate_client.schema.get_class_shards("Document")[0]["name"]

    assert new_shard_name == old_shard_name


def test_delete(embeddings, weaviate_client):

    embeddings.index([(0, "Lorem ipsum", None)])
    objects = weaviate_client.data_object.get(class_name="Document")["objects"]

    assert len(objects) == 1

    embeddings.delete([0])
    objects = weaviate_client.data_object.get(class_name="Document")["objects"]

    assert len(objects) == 0


def test_client_batch_config(weaviate_db):
    config = {
        "weaviate": {
            "url": WEAVIATE_DB_URL,
            "batch": {
                "batch_size": 64,
                "dynamic": True,
            },
        }
    }

    backend = ann.Weaviate(config)

    assert backend.client.batch._num_workers == 1
    assert backend.client.batch._connection_error_retries == 3
    assert backend.client.batch._batch_size == 64
    assert backend.client.batch.dynamic == True


def test_index_exists(embeddings, weaviate_client):

    embeddings.index([(0, "Lorem ipsum", None)])
    assert embeddings.count() == 1

    weaviate_client.schema.delete_class("Document")
    with pytest.raises(weaviate.exceptions.UnexpectedStatusCodeException):
        embeddings.count()


def test_normalize_cosine_distance():
    assert ann.normalize_cosine_distance(0.0) == 1.0
    assert ann.normalize_cosine_distance(2.0) == -1.0


def test_upsert(embeddings, weaviate_client):
    data = [
        "US tops 5 million confirmed virus cases",
        "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
        "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
        "The National Park Service warns against sacrificing slower friends in a bear attack",
        "Maine man wins $1M from $25 lottery ticket",
        "Make huge profits without work, earn up to $100,000 a day",
    ]
    embeddings.index([(uid, text, None) for uid, text in enumerate(data)])

    udata = data.copy()

    udata[0] = "See it: baby panda born"
    embeddings.upsert([(0, udata[0], None)])

    old_uid = embeddings.search("feel good story", 1)[0][0]

    embeddings.delete([0])

    new_uid = embeddings.search("feel good story", 1)[0][0]

    assert old_uid == new_uid

def test_upsert_with_new_embeddings(weaviate_db, weaviate_client, tmp_path):
    savefile = str(tmp_path / "test")

    old_embeddings = Embeddings(
        {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "backend": "weaviate_txtai.ann.weaviate.Weaviate",
            "weaviate": {"url": WEAVIATE_DB_URL, "overwrite_index": False},
        }
    )

    data = [
        "US tops 5 million confirmed virus cases",
        "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
        "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
        "The National Park Service warns against sacrificing slower friends in a bear attack",
        "Maine man wins $1M from $25 lottery ticket",
        "Make huge profits without work, earn up to $100,000 a day",
    ]

    old_embeddings.index([(uid, text, None) for uid, text in enumerate(data)])

    old_embeddings.save(savefile)

    new_embeddings = Embeddings()
    new_embeddings.load(savefile)

    udata = data.copy()

    udata[0] = "See it: baby panda born"
    new_embeddings.upsert([(0, udata[0], None)])

    old_uid = new_embeddings.search("feel good story", 1)[0][0]

    new_embeddings.delete([0])

    new_uid = new_embeddings.search("feel good story", 1)[0][0]

    assert old_uid == new_uid