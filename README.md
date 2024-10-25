# llama-relik

## Entity Linking and Relationship Extraction With Relik for creating a Knowledge Graph in Neo4J with LlamaIndex and LLM GraphRAG

Changed the [llama_relik.ipynb](https://github.com/tomasonjo/blogs/blob/master/llm/llama_relik.ipynb) notebook from Tomaz Bratanic's Neo4j [blog article](https://neo4j.com/developer-blog/entity-linking-relationship-extraction-relik-llamaindex/) to use fastcoref instead of coreferee.

Fastcoref was mentioned in the [medium article](https://medium.com/neo4j/entity-linking-and-relationship-extraction-with-relik-in-llamaindex-ca18892c169f) version of the Neo4j blog article in the comments.

Fastcoref is supposed to be better for LLM genAI accuracy as covered in Michael Wood's comments and other [medium article](https://medium.com/@michaelwood33311/creating-accurate-ai-coreference-resolution-with-fastcoref-20f06044bdf9).

Fastcoreref takes more time with Colab (CPU or free T4 GPU runtime type) than coreferee. Wasn't bad on a local machine with python file (llama-relik-public.py) instead of the Jupyter notebook.
Also don't have the coreferee problem of having to rebuild from source / spacy version issue.

This llama-relik-fastcoref.ipynb can be open and run with [Google Colab](https://colab.research.google.com/) after setting up a free account.
With Google Colab you don't have to worry about your python and machine environment.

If using the notebook, you need to add your Neo4j database info and your OpenAI key to the notebook file.
If changes are made in colab, you can be save locally with File/Download/.ipynb

If using the llama-relik-public.py python file, you also need to add your Neo4j database info and your OpenAI key.
With the python file, you have to setup your python environment (use a virtual env) with the packages mentioned in the comments at the top.

Currently, Relik has an issue on Windows, and also is not enabled for python 3.12. (Colab runs Ubuntu and was running python 3.10).
I submitted a Relik pull request (was pulled in) that fixed the runtime issue with Windows. This was after Relik 1.0.7 .
I submitted a second Relik pull request (hasn't been pulled in yet) that fixes being able to build from source on Windows and re-allows python 3.12 .
The faiss-gpu is only available on conda for linux-64. The faiss-cpu package is available for win-64, osx-arm64, and linux-64.
Based on this, I seemed only to be able to use Relik models with "cpu", not "cuda" on Windows.

I was seeing llama-relik-public.py running about 8 times faster on Ubuntu than Windows. On Ubuntu, the time for gpu "cuda" vs. "cpu" were about the same.
Time was about 1 minute total on Ubuntu vs 8 minutes on Windows (for "cpu"). This testing was before
the python file was changed from running 10 small news articles to 1 space station sample text.

Maybe would be better in a docker on Windows running on a Linux WSL2.



