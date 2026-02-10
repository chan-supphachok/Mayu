from openai import OpenAI
import re
import pandas as pd
from langchain_core.output_parsers.json import JsonOutputParser
from fuzzywuzzy import fuzz
import yaml

parser = JsonOutputParser()
def loadconfig(file_path):
    """
    Loads a YAML configuration file and returns it as a dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
config = loadconfig("config.yaml")


def get_response(system, user):
    
    client = OpenAI(
    base_url=config["base_url"],
    api_key=config["api_key"],
)
    completion = client.chat.completions.create(
        model="Qwen/Qwen3-14B",
        messages=[
            {"role":"system", "content": system },
            {"role": "user", "content": user},
        ],
    )
    return completion.choices[0].message.content.split("</think>")[-1]

def parser_extract(text):
    pattern = r"json\s*(\{.*?\})\s*"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_block = match.group(1)

        return parser.parse(json_block)
    else:
        return None

def search_mem(df: pd.DataFrame, query_text, topn: int = 10) -> str: # get top 10 memory
    if df.empty:
        return ""
    df['similarity'] = df['short_main_idea'].apply(lambda x: fuzz.ratio(x, query_text))

    # Sort by similarity
    top = df.sort_values(by='similarity', ascending=False).head(topn).to_string(index=False)
    #print(df_sorted)
    return top