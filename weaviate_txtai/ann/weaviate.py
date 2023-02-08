import uuid

import weaviate
from txtai.ann import ANN
from weaviate import Client
from weaviate.util import generate_uuid5

DEFAULT_SCHEMA = {
    "class": "Document",
    "properties": [{"name": "docid", "dataType": ["int"]}],
    "vectorIndexConfig": {"distance": "cosine"},
}


class Weaviate(ANN):
    """
    Builds an ANN index using the Faiss Library
    """

    def __init__(self, config):
        super().__init__(config)

        self.weaviate_config = self.config.get("weaviate", {})
        url = self.weaviate_config.get("url", "http://localhost:8080")
        self.client = Client(url)

        self.config["offset"] = 0
        self._create_schema()
        self._configure_client()

    def _configure_client(self):
        self.client.batch.configure(batch_size=100, num_workers=1)

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

        self.client.schema.create_class(schema)

    def index(self, embeddings):

        with self.client.batch as batch:
            for embedding in embeddings:
                random_identifier = uuid.uuid4()
                object_uuid = generate_uuid5(random_identifier)
                batch.add_data_object(
                    data_object={
                        "docid": self.config["offset"],
                    },
                    class_name="Document",
                    vector=embedding,
                    uuid=object_uuid,
                )

                self.config["offset"] += 1

    def search(self, queries, limit):

        nearVector = {"vector": queries[0]}

        results = (
            self.client.query.get("Document", properties=["docid"])
            .with_additional("distance")
            .with_near_vector(nearVector)
            .with_limit(limit)
            .do()
        )

        results = results["data"]["Get"]["Document"]

        return [
            [(result["docid"], result["_additional"]["distance"]) for result in results]
        ]
