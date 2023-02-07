import uuid

from txtai.ann import ANN
from weaviate import Client
from weaviate.util import generate_uuid5


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

    def _create_schema(self):
        schema = self.weaviate_config.get("schema", {})
        if schema:
            self.client.schema.create_class(schema)
        else:
            schema = {
                "class": "Document",
                "properties": [{"name": "text", "dataType": ["text"]}],
                "vectorIndexConfig": {"distance": "cosine"},
            }

            self.client.schema.create_class(schema)

    def index(self, embeddings):

        with self.client.batch as batch:
            for embedding in embeddings:
                random_identifier = uuid.uuid4()
                object_uuid = generate_uuid5(random_identifier)
                batch.add_data_object(
                    data_object={},
                    class_name="Document",
                    vector=embedding,
                    uuid=object_uuid,
                )

                self.config["offset"] += 1

    def search(self, queries, limit):

        nearVector = {"vector": queries[0]}

        results = (
            self.client.query.get("Document")
            .with_additional("distance")
            .with_near_vector(nearVector)
            .with_limit(limit)
            .do()
        )

        results = results["data"]["Get"]["Document"]

        return [
            [(i, result["_additional"]["distance"]) for i, result in enumerate(results)]
        ]
