import os
import pandas as pd
from dotenv import load_dotenv,find_dotenv
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.llms.anthropic import Anthropic
from prompts import instruction_str, new_prompt, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pdf import canada_engine

load_dotenv(find_dotenv())


llm = Anthropic(model="claude-2.0", temperature=0.1)

population_path = os.path.join("data", "population.csv")
population_df  = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str, llm=llm
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})


tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine, 
        metadata=ToolMetadata(
            name="population_data",
            description="this gives information at the world population and demographics",
        ),
    ),
    QueryEngineTool(
        query_engine=canada_engine,
        metadata=ToolMetadata(
            name="canada_data",
            description="this gives detailed information about canada the country.",
        ),
    ),
]

agent = ReActAgent.from_tools(tools, llm=llm,verbose=True, context=context)

while(prompt := input("Enter a prompt (q to quit): ")) !="q":
    result = agent.query(prompt)
    print(result)