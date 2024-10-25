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

text = '''The International Space Station (ISS) is an awe-inspiring feat 
of human engineering and collaboration. Orbiting the Earth at an 
altitude of approximately 408 kilometers, the ISS serves as a research 
laboratory and living space for astronauts from around the world. 
NASA, in partnership with other space agencies such as Roscosmos, 
ESA, JAXA, and CSA, has been instrumental in the construction and 
operation of this remarkable structure. The ISS has facilitated 
groundbreaking scientific experiments and discoveries in various fields, 
including physics, biology, and astronomy. Notable astronauts like 
Chris Hadfield and Scott Kelly have spent significant time aboard 
the ISS, conducting experiments and gathering data to expand our 
knowledge of space exploration. However, with the advancement of 
commercial space travel, companies like SpaceX and Blue Origin are 
aiming to make space more accessible and potentially challenge the 
ISS's monopoly on human space presence.'''

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

    #NUMBER_OF_ARTICLES = 1
    #news = pd.read_csv("https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv")
    #news = news.head(NUMBER_OF_ARTICLES)

    print("start coref")

    #news["coref_text"] = news["text"].apply(coref_text)
    #documents = [
    #    Document(text=f"{row['title']}: {row['coref_text']}") for i, row in news.iterrows()
    #]
    #print(news['text'][1])
    #print(news['coref_text'][1])

    coref = coref_text(text)
    document = Document(text=f"{'title'}: {coref}")
    documents = []
    documents.append(document)

    print("start relik")
    #relik = RelikPathExtractor(
    #    model="relik-ie/relik-relation-extraction-small", model_config={"skip_metadata": True}
    #)

    # Use on Pro Collab with GPU or high end local machine / GPU
    relik = RelikPathExtractor(
        #model="relik-ie/relik-cie-small", model_config={"skip_metadata": True, "device":"cpu"}
        #model="relik-ie/relik-cie-xl", model_config={"skip_metadata": True, "device":"cpu"}
        #model="relik-ie/relik-relation-extraction-small", model_config={"skip_metadata": True, "device":"cpu"}
        model="relik-ie/relik-relation-extraction-large", model_config={"skip_metadata": True, "device":"cpu"}
        #model="relik-ie/relik-relation-extraction-large", model_config={"skip_metadata": True, "device":"cuda"}
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
    #response = query_engine.query("What happened at Ryanair?")
    response = query_engine.query("Who has been on the ISS?")

    print(str(response))


if __name__ == "__main__":
    #freeze_support()
    main()

