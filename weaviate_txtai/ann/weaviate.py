from weaviate import Client

from txtai.ann import ANN


class Weaviate(ANN):
    """
    Builds an ANN index using the Faiss Library
    """

    def __init__(self, config):
        super().__init__(config)