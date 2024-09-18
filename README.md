# llama-relik

## Entity Linking and Relationship Extraction With Relik for creating a Knowledge Graph in Neo4J with LlamaIndex and LLM GraphRAG

Changed the [llama_relik.ipynb](https://github.com/tomasonjo/blogs/blob/master/llm/llama_relik.ipynb) notebook from Tomaz Bratanic's Neo4j [blog article](https://neo4j.com/developer-blog/entity-linking-relationship-extraction-relik-llamaindex/) to use fastcoref instead of coreferee.

Fastcoref was mentioned in the [medium article](https://medium.com/neo4j/entity-linking-and-relationship-extraction-with-relik-in-llamaindex-ca18892c169f) version of the Neo4j blog article in the comments.

Fastcoref is supposed to be better for LLM genAI accuracy as covered in Michael Wood's comments and other [medium article](https://medium.com/@michaelwood33311/creating-accurate-ai-coreference-resolution-with-fastcoref-20f06044bdf9).

Fastcoreref takes more time with Colab (CPU or free T4 GPU runtime type) than coreferee. Wasn't bad on a local machine (Ubuntu 22.04 / AMD 5950x CPU / 3090 GPU).
Also don't have the coreferee problem of having to rebuild from source / spacy version issue.

This llama-relik-fastcoref.ipynb can be open and run with [Google Colab](https://colab.research.google.com/) after setting up a free account.
With Google Colab you don't have to worry about your python and machine environment.
Currently Relik has an issue on Windows, and also is not enabled for python 3.12 (Colab was running python 3.10).

You need to add your Neo4j database info and your OpenAI key.
Changes in colab can be save locally with File/Download/.ipynb

