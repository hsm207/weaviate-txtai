"""
Weaviate client module
Based on https://github.com/neuml/txtai.weaviate
"""

import weaviate

from txtai.pipeline import Pipeline


class Weaviate(Pipeline):
    """
    Weaviate pipeline client. Supports indexing and searching content with Weaviate.
    """

    def __init__(self, url="http://localhost:8080", custom_schema=None):
        """
        Create a new client.

        Args:
            url: Weaviate service url
            custom_schema: A schema for one class in Weaviate
        """

        self.class_name = None
        self.content_field = "content"

        self.client = weaviate.Client(url)
        self._create_schema(custom_schema)

    def _create_schema(self, schema):
        if not schema:
            schema = {
                "class": "Document",
                "description": "A document class to store text",
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "A text snippet",
                    },
                    {
                        "name": "source",
                        "dataType": ["string"],
                        "description": "The text snippet's source e.g. filename, url, etc.",
                    },
                ],
                "vectorIndexConfig": {"distance": "cosine"},
            }

        self.class_name = schema["class"]

        assert any(
            p["name"] == "content" and p["dataType"][0] == "text"
            for p in schema["properties"]
        ), "Custom schema must have a property named 'content' with type 'text'!"

        self.client.schema.create_class(schema)

    def __call__(self, inputs, action="index"):
        """
        Executes an action with Weaviate.

        Args:
            inputs: data inputs
            action: action to perform - index or search

        Returns:
            results
        """

        if action == "index":
            return [self.index(data, vector) for data, vector in inputs]

        # Default to search action
        return [self.search(vector) for vector in inputs]

    def index(self, data, vector):
        """
        Indexes data-vector pair in Weaviate.

        Args:
            data: record metadata
            vector: record embeddings

        Returns:
            uuid from Weaviate
        """

        return self.client.data_object.create(
            {"content": data},
            self.class_name,
            vector=vector,
        )

    def search(self, vector):
        """
        Runs a search using input vector.

        Args:
            vector: input vector

        Returns:
            search results
        """

        nearvector = {"vector": vector}
        return (
            self.client.query.get(
                self.class_name, [self.content_field, "_additional {certainty}"]
            )
            .with_near_vector(nearvector)
            .with_limit(1)
            .do()
        )
