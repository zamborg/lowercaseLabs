from collections import abc
import time
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from typing import Any
import openai
import tiktoken
import os

class BaseConfig:
    def __init__(self, **kwargs):
        self.config = {}
        self.update(**kwargs)

    def update(self, **kwargs):
        self.config.update(kwargs)

    def __getattr__(self, attr):
        return self.config.get(attr, None)
    
class Config(BaseConfig):
    """
    Override default
    """
    default = {}
    def __init__(self, **kwargs):
        super().__init__(
            **{
                **self.default, 
                **kwargs
            }
        )

class AgentDescription():
    def __init__(self, name, description, api) -> None:
        """
        AgentDescription is a class to hold agent descriptions
        """
        self.name = name
        self.description = description
        self.api = api
    
    def __str__(self) -> str:
        ret = "{"
        for k,v in self._dict.items():
            ret += f"\n    {k} : {v},"
        ret += "\n}"
        return ret

    @property
    def _dict(self) -> dict:
        return {
            "name" : self.name,
            "description" : self.description,
            "api" : self.api
        }


class Agent():
    def __init__(self, config: Config, description: AgentDescription) -> None:
        # TODO: refactor so it just uses the self.description value set by each subsequently inherited agent
        self.configuration = config
        self.description = description

    def go(self, **kwargs):
        """
        context: Dict with keys/values
        """
        raise NotImplementedError("This must be overridden")
    
    def __call__(self, **kwargs) -> Any:
        return self.go(**kwargs)
    
    def __getattr__(self, attr):
        return self.configuration.__getattr__(attr)
    
    def __str__(self) -> str:
        return str(self.description)


class AgentDescriptionList():
    def __init__(self, **kwargs) -> None:
        """
        keyword arguments with agent_name : instantiated agent
        """
        self.agents = kwargs

    def __str__(self) -> str:
        return "\n".join([str(v) + "," for v in self.agents.values()])
        

class SearchConfig(Config):
    default = {
        "k" : 5, # default 5 search results
        "region" : "us-en",
        "safesearch" : "Off",
        "backend" : "api", # api, html, or lite
        "timeout" : "y",
        "wait" : 0.75 # wait 0.75 seconds between calls
    }


class SearchResult():

    @staticmethod
    def get_url(url):
        return BeautifulSoup(requests.get(url).text, features="lxml").text
    
    def __init__(self, result) -> None:
        """
        result is a dict with keys title, href, and body
        """
        self.title = result['title']
        self.url = result['href']
        # drop the body and reget it
        self.body = SearchResult.get_url(self.url)

    @property
    def text(self):
        text = f"TITLE: {self.title}\n\n"
        text += f"BODY:\n" + self.body + "\n"
        return text


class SearchAgent(Agent):

    DESCRIPTION = AgentDescription(
        name= "Search Agent",
        description= "Search Agent is an agent that accepts a QUERY_TEXT and K_VALUE as an input and returns the string content of the first K_VALUE between (0-10) articles according to the google search of the QUERY_TEXT",
        api='{"query" : QUERY_TEXT, "k" : K_VALUE}'
    )
    
    def __init__(self, config: Config) -> None:
        super().__init__(config, SearchAgent.DESCRIPTION)
        self.searcher = DDGS()

    def go(self, query, k=None):
        if k is None:
            k = self.config.k
        results = self.search(query, k)
        
        return [SearchResult(r) for r in results]
    

    # Use the duckduckgo_search library
    def search(self, query, k):
        # api reference: https://github.com/deedy5/duckduckgo_search#1-text---text-search-by-duckduckgocom
        gen = self.searcher.text(
            query,
            region = self.region,
            safesearch = self.safesearch,
            timelimit = self.timelimit,
            backend = self.backend,
        )
        res = []
        for i in range(k):
            res.append(next(gen))
            time.sleep(self.wait)
        return res
        

class OpenAIAgent:
    """
    Class to cointain our openAI inference logic
    """
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    BUFFER = 10 # 10 token buffer
    def __init__(self, tokenizer="cl100k_base", max_tokens=4000, **kwargs) -> None:
        openai.api_key= os.environ['OPENAI_API_KEY']
        self.messages, self.cache = [], []
        self.tokenizer=tiktoken.get_encoding(tokenizer)
        self.max_tokens = max_tokens

    @staticmethod
    def create_message(role, content):
        """
        Creates a message
        """
        return {"role": role, "content": content}
    
    @property
    def headroom(self):
        """
        updates self.headroom
        """
        return self.max_tokens-len(self)
    
    def chunk_content(self, content, headroom = None):
        # returns a generator that chunks content into self.headroom size:
        tokenized = self.tokenizer.encode(content)
        size = min(self.max_len, len(tokenized)) if self.max_len is not None else len(tokenized)
        step = self.headroom-self.BUFFER if headroom is None else headroom
        step = min(step, size)
        for i in range(0, size, step):
            yield self.tokenizer.decode(tokenized[i: i+step])


    def _add_message(self, role, content):
        """
        Add a message to the call
        """
        self.messages.append({"role": role, "content": content})
    
    def add_system(self, content):
        """
        Add a system message to the message stack
        """
        self._add_message(OpenAIAgent.SYSTEM, content)

    def add_user(self, content):
        """
        add a user message to the message stack
        """
        self._add_message(OpenAIAgent.USER, content)

    def add_assistant(self, content):
        """
        add an agent message to the message stack
        """
        self._add_message(OpenAIAgent.ASSISTANT, content)

    def infer(self):
        raise NotImplementedError("infer must be Overridden by a subclass")
    
    def __len__(self):
        return len(self.tokenizer.encode(str(self.messages)))