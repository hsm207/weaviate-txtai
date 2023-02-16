import uuid

import weaviate
from txtai.ann import ANN
from weaviate import Client
from weaviate.util import generate_uuid5
import logging

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)


DEFAULT_SCHEMA = {
    "class": "Document",
    "properties": [{"name": "docid", "dataType": ["int"]}],
    "vectorIndexConfig": {"distance": "cosine"},
}

DEFAULT_BATCH_CONFIG = {
    "batch_size": None,
    "creation_time": None,
    "timeout_retries": 3,
    "connection_error_retries": 3,
    "weaviate_error_retries": None,
    "callback": None,
    "dynamic": False,
    "num_workers": 1,
}


def check_index_exists(func):
    def wrapper(self, *args, **kwargs):
        try:
            self.client.schema.get(self.index_name)
            return func(self, *args, **kwargs)
        except weaviate.exceptions.UnexpectedStatusCodeException:
            logger.error(
                f'Unable to find index "{self.index_name}" in weaviate. Aborting call to {func.__name__}'
            )
            raise

    return wrapper


class Weaviate(ANN):
    """
    Builds an ANN index using the Weaviate vector search engine
    """

    def __init__(self, config):
        super().__init__(config)

        self.weaviate_config = self.config.get("weaviate", {})
        url = self.weaviate_config.get("url", "http://localhost:8080")
        self.client = Client(url)

        self.config["offset"] = 0
        self.overwrite_index = self.weaviate_config.get("overwrite_index", True)
        self.index_name = None
        self._create_schema()

        batch_config = self.weaviate_config.get("batch", DEFAULT_BATCH_CONFIG)
        self._configure_client(**batch_config)

    def _configure_client(
        self,
        batch_size=None,
        creation_time=None,
        timeout_retries=3,
        connection_error_retries=3,
        weaviate_error_retries=None,
        callback=None,
        dynamic=False,
        num_workers=1,
    ):
        self.client.batch.configure(
            batch_size,
            creation_time,
            timeout_retries,
            connection_error_retries,
            weaviate_error_retries,
            callback,
            dynamic,
            num_workers,
        )

    def _is_valid_schema(self, schema):

        docid_key = "docid"
        docid_type = "int"
        properties = schema["properties"]

        weaviate.schema.validate_schema.check_class(schema)

        for prop in properties:
            if prop["name"] == docid_key and prop["dataType"][0] == docid_type:
                return True

        return False

    def _create_schema(self):
        schema = self.weaviate_config.get("schema", DEFAULT_SCHEMA)

        if not self._is_valid_schema(schema):
            raise weaviate.exceptions.SchemaValidationException(
                f"Class {schema['class']} must have a property named 'docid' of type 'int'"
            )

        if self.client.schema.contains(schema):
            if not self.overwrite_index:
                raise weaviate.exceptions.ObjectAlreadyExistsException(
                    f"Index {schema['class']} already exists"
                )
            else:
                self.client.schema.delete_class(schema["class"])

        self.client.schema.create_class(schema)
        self.index_name = schema["class"]

    def index(self, embeddings):
        self.append(embeddings)

    def append(self, embeddings):

        with self.client.batch as batch:
            for embedding in embeddings:
                random_identifier = uuid.uuid4()
                object_uuid = generate_uuid5(random_identifier)
                batch.add_data_object(
                    data_object={
                        "docid": self.config["offset"],
                    },
                    class_name=self.index_name,
                    vector=embedding,
                    uuid=object_uuid,
                )

                self.config["offset"] += 1

    def _get_uuid_from_docid(self, docid):

        results = (
            self.client.query.get(self.index_name)
            .with_additional("id")
            .with_where(
                {
                    "path": ["docid"],
                    "operator": "Equal",
                    "valueInt": docid,
                }
            )
            .do()
        )

        return results["data"]["Get"][self.index_name][0]["_additional"]["id"]

    @check_index_exists
    def delete(self, ids):

        for id in ids:
            # TODO: rewrite when weaviate supports IN operator
            #       See: https://github.com/weaviate/weaviate/issues/2387
            uuid = self._get_uuid_from_docid(id)
            self.client.data_object.delete(uuid, class_name=self.index_name)

    @check_index_exists
    def search(self, queries, limit):

        nearVector = {"vector": queries[0]}

        # use distance to score similarity
        # TODO: discuss if this is the best way to do it
        #       in txtai, higher similarity score means more similar
        #       but in weaviate, lower distance means more similar
        results = (
            self.client.query.get(self.index_name, properties=["docid"])
            .with_additional("distance")
            .with_near_vector(nearVector)
            .with_limit(limit)
            .do()
        )

        results = results["data"]["Get"][self.index_name]

        return [
            [(result["docid"], result["_additional"]["distance"]) for result in results]
        ]

    @check_index_exists
    def count(self):
        results = self.client.query.aggregate(self.index_name).with_meta_count().do()
        return results["data"]["Aggregate"][self.index_name][0]["meta"]["count"]

    def save(self, path):
        raise NotImplementedError(
            """
            Saving the index through txtai is not yet supported for the Weaviate backend.
            Please use Weaviate's API instead.
            See: https://weaviate.io/developers/weaviate/configuration/backups#introduction
            """
        )

    def load(self, path):
        raise NotImplementedError(
            """
            Loading the index through txtai is not yet supported for the Weaviate backend.
            Please use Weaviate's API instead.
            See: https://weaviate.io/developers/weaviate/configuration/backups#introduction
            """
        )
