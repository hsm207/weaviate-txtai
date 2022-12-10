# Introduction

This package aims to provide a convenient way to use [weaviate](https://github.com/semi-technologies/weaviate) as a document store when building solutions using [txtai](https://github.com/neuml/txtai).

# Installation
```bash
pip install weaviate-txtai
```
# Usage

Here is an example of a txtai configuration that uses this package for semantic search:

```yaml
embeddings:
  path: sentence-transformers/nli-mpnet-base-v2

nop:

weaviate_txtai.client.Weaviate:
  url: http://weaviate:8080
  custom_schema:
      class: "Post"
      properties:
       - name: "content"
         dataType:
          - text
      vectorIndexConfig:
       distance: "dot"

workflow:
  index:
    batch: 2500
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
```

Read this [blog post](https://medium.com/@_init_/how-to-quickly-build-a-semantic-search-system-with-txtai-and-weaviate-fd4084e93aaa) for an end-to-end example.

# Roadmap
    ¯\_(ツ)_/¯
# TODOs
- [ ] Assess community interest in this package
- [ ] Figure out roadmap to 1.0.0 release
- [ ] Write docs for contributing
