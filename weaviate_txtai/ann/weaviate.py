from weaviate import Client

from txtai.ann import ANN


class Weaviate(ANN):
    """
    Builds an ANN index using the Faiss Library
    """

    def __init__(self, config):
        super().__init__(config)

        self.weaviate_config = self.config.get("weaviate", {})
        url = self.weaviate_config.get("url", "http://localhost:8080")
        self.client = Client(url)

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
