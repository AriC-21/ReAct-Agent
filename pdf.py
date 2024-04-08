import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

def get_index(data, index_name):
    index  = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )
    return index

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
llm = Anthropic(model="claude-2.0", temperature=0.1)
pdf_path = os.path.join("data", "Canada.pdf")
canada_pdf = SimpleDirectoryReader(
        input_files=[pdf_path]
    ).load_data()
canada_index = get_index(canada_pdf,"canada")
canada_engine = canada_index.as_query_engine(llm=llm)