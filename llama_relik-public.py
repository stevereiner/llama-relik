#pip install fastcoref spacy
#pip install llama-index-extractors-relik llama-index-graph-stores-neo4j llama-index-llms-openai llama-index
#python -m spacy download en_core_web_lg

import nest_asyncio
from llama_index.graph_stores.neo4j import Neo4jPGStore
import pandas as pd
from fastcoref import spacy_component
import spacy
from llama_index.core import Document
from datetime import datetime
import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import PropertyGraphIndex
from llama_index.extractors.relik.base import RelikPathExtractor
from multiprocessing import freeze_support

os.environ["OPENAI_API_KEY"] = "sk-your-openai-key"
username="neo4j"
password="your neo4j password"
url="neo4j+s://your-aura-instance-id.databases.neo4j.io"

nest_asyncio.apply()

coref_nlp = spacy.load('en_core_web_lg')
coref_nlp.add_pipe('fastcoref')

def coref_text(text):
    coref_doc = coref_nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})
    resolved_text = coref_doc._.resolved_text
    return resolved_text

def main():

    graph_store = Neo4jPGStore(
        username=username,
        password=password,
        url=url,
        refresh_schema=False
    )

    NUMBER_OF_ARTICLES = 10
    news = pd.read_csv("https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv")
    news = news.head(NUMBER_OF_ARTICLES)

    print("start coref")
    news["coref_text"] = news["text"].apply(coref_text)
    documents = [
        Document(text=f"{row['title']}: {row['coref_text']}") for i, row in news.iterrows()
    ]

    #print(coref_text(news['text'][5]))

    print("start relik")
    #relik = RelikPathExtractor(
    #    model="relik-ie/relik-relation-extraction-small", model_config={"skip_metadata": True}
    #)

    # Use on Pro Collab with GPU or high end local machine / GPU
    relik = RelikPathExtractor(
    #   model="relik-ie/relik-cie-small", model_config={"skip_metadata": True, "device":"cuda"}
    #   model="relik-ie/relik-relation-extraction-small", model_config={"skip_metadata": True, "device":"cuda"}
        model="relik-ie/relik-cie-small", model_config={"skip_metadata": True, "device":"cpu"}
    )

    print("start openai")
    llm = OpenAI(model="gpt-4o", temperature=0.0)
    embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

    print("start PropertyGraphIndex") 
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[relik],
        llm=llm,
        embed_model=embed_model,
        property_graph_store=graph_store,
        show_progress=True,
    )

    print("start index.as_query_engine")
    query_engine = index.as_query_engine(include_text=True)

    print("start query")
    response = query_engine.query("What happened at Ryanair?")

    print(str(response))


if __name__ == "__main__":
    #freeze_support()
    main()

