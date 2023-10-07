import os
from typing import Any
import openai
import tiktoken
from agents import *
from agents import AgentDescription, Config
import json

class PersistentChat():
    """
    Persistent Chat Agent

    PersistentChat calls openai to construct 
    """
    def __init__(self) -> None:
        pass
    
AGENTS = AgentDescriptionList(
    SearchAgent = SearchAgent(SearchConfig())
)

class Selector(OpenAIAgent):
    """
    Orchestrator determines what function to call out of the set of options
    """
    def __init__(self, agents: AgentDescriptionList, **kwargs) -> None:
        super().__init__(**kwargs)
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.add_system("You are an assistant orchestrator. Provided with a list of agent names, descriptions, and an API to interact with them you will select the best agent based on user question and description. You will then output a json of the following format: { name : AGENT_NAME, api : API} according the the respective agent description.\n")
        self.add_system("Here are the agents you can select amongst:")
        self.add_system(str(agents))
        self.add_system("Return ONLY properly formatted JSON Content")
        self.add_assistant("Thank you. I will return properly formatted JSON that includes the name and API of the most suitable model for the user query.")

    def infer(self, query):
        res = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = self.messages + [OpenAIAgent.create_message(OpenAIAgent.USER, query)]
        )
        return json.loads(res["choices"][0]["message"]["content"])


class Condensor(OpenAIAgent):
    MAX_LEN = 10000 # only condense 10k tokens
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_system("You are an information compression agent. The user will provide you a large piece of content, you will return a condensed form of the content while retaining all pertinent information.")
        self.add_assistant("I understand. I will condense user provided content to the smallest possible portion while retaining all relevant information.")
        self.max_len = self.MAX_LEN if kwargs.get("max_len", None) is None else kwargs.get("max_len")


    def infer(self, text: str):
        """
        TODO: add query to infer call for contextual information.
        """
        res = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = self.messages + [OpenAIAgent.create_message(OpenAIAgent.USER, text)]
        )
        return res["choices"][0]["message"]["content"]
    
    def condense(self, context):
        # use the self.chunk_content method
        # TODO: refactor this to include the previously condensed message
        condensed = ""
        for chunk in self.chunk_content(context, self.headroom):
            condensed += self.infer(chunk)

        return condensed
    
    def __call__(self, context) -> Any:
        return self.condense(context)
    

class QueryRewriter(OpenAIAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.add_system("You are a query rewriting agent. A user will ask you a query that you might not have enough information to answer. Rewrite the query for an agent that has the relevant context to answer the user's question")
        self.add_system("You are a query rewriting agent. You will take a user query, and rewrite it for an ANSWER_AGENT. ANSWER_AGENT has access to the internet and has already ingested relevant information. Assume that ANSWER_AGENT has sufficient context, rewrite the user query so ANSWER_AGENT can best respond")
        self.add_assistant("I am a query rewriting agent. I will take a query from a user and rewrite it such that the ANSWER_AGENT will be able to sufficiently respond")
        self.add_user(
            """Now that you understand your role. I will give you an example for how to respond:
            User: Summarize the most recent engadget article
            YOU: Summarize this article for me, and provide me with all important information.
            """
        )
        self.add_assistant("I understand. I will rewrite user queries in the style of your example.")

    def infer(self, query):
        messages = self.messages + [OpenAIAgent.create_message(OpenAIAgent.USER, query)]
        res = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = messages
        )
        return res["choices"][0]["message"]["content"]
        

class ChatConfig(Config):
    default = {
        "store" : True, 
        "compress" : True,
        "verbose" : True,
        "max_len" : 5000 # max tokens to condense
    }
 
class ChatAgent(OpenAIAgent, Agent):
    # TODO: FIND A WAY TO REWRITE QUERIES

    def __init__(self, config, **kwargs) -> None:
        Agent.__init__(self, config, AgentDescription("", "", ""))
        OpenAIAgent.__init__(self, **kwargs)
        self.search = SearchAgent(SearchConfig())
        self.selector = Selector(AgentDescriptionList(SearchAgent = self.search))
        self.query_rewrite = QueryRewriter()
        if self.compress:
            self.condenser = Condensor(max_len = self.max_len)
        self.add_system("You are a chat assistant. The user will ask you a query, and if you need more information you will request it. The user will then provide you more information, and you are to answer their original query. The user provided information will be sufficient to answer the query.\n")
        self.add_assistant("I understand. I will aks the user for relevant information to answer their query, then provide an answer.\n")
        # self.add_user("Remember, while you may not have access to the internet, and my query may seem like you need more information. All the information you need to answer my query is in the user provided context block.")
        # self.add_assistant("I understand, I will answer your query using the information in the user provided context block.")
        self.add_user("CONTEXT:\n")
    
    def infer(self, query):
        # todo save a self.cache instead of selfing multiple variables
        search_query = self.selector.infer(query=query)
        if self.verbose:
            print(search_query)
        context = self.search(**search_query["api"]) # TODO: un-hardcode this
        # TODO add k_chunking to the condenser
        condensed = self.condenser("\n".join([c.text for c in context]))

        if self.verbose:
            for c in condensed:
                print(c)
        final_query = self.query_rewrite.infer(query)

        if self.verbose:
            print(final_query)

        final_messages = self.messages + [OpenAIAgent.create_message(OpenAIAgent.USER, f"My question is: {final_query}")]
        final_messages.append(OpenAIAgent.create_message(OpenAIAgent.ASSISTANT, "Could you please provide me with the context to answer this question."))
        final_messages.append(OpenAIAgent.create_message(OpenAIAgent.USER, f"Here is all the information you need:\n{condensed}"))

        # messages = self.messages + [OpenAIAgent.create_message(OpenAIAgent.SYSTEM, condensed)]
        # messages.append(OpenAIAgent.create_message(OpenAIAgent.USER, final_query))
        res = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = final_messages
        )

        if self.store:
            self.cache = {
                "search_query" : search_query,
                "condensed" : condensed,
                "final_query" : final_query,
                "final_messages" : final_messages
            }

        return res["choices"][0]["message"]["content"]
        
class EmailAgent(OpenAIAgent, Agent):
    # TODO: FIND A WAY TO REWRITE QUERIES

    def __init__(self, config, **kwargs) -> None:
        Agent.__init__(self, config, AgentDescription("", "", ""))
        OpenAIAgent.__init__(self, **kwargs)