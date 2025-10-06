import json
from constants.config import *
from constants.prompts import *
from constants.data import *
from constants.helpers import *
from constants.retriever_type import *
from langchain_core.prompts import PromptTemplate
from langchain.callbacks import StdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.globals import set_verbose
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
set_verbose(True)

class Rewriter():
    def __init__(self, memory,llm):
        '''Get rewriter model and pipeline'''
        # rewriter_llm_pipeline = get_llm_pipeline(get_model(rewriter_model_path, rewriter_model_id), rewriter_model_path)
        self.llm = llm
        self.memory = memory
        '''Create the rewriter chain'''
        self.query_rewrite_chain = self.get_query_rewrite_chain(llm, memory)
        
    def get_query_rewrite_chain(self,llm, memory):
        '''Get the Query rewriting chain with the loaded LLM pipeline'''
        print("Function: get_query_rewrite_chain \t Status: Started")
        #create a chain to rewrite the query using the chat_history and question from memory using the llm
        query_rewrite_prompt = PromptTemplate(template=get_prompt(type="Rewrite"), input_variables=['chat_history', 'question'])
        query_rewrite_chain = ({
        "question": RunnablePassthrough(),
        "chat_history": lambda x : memory.load_memory_variables({"chat_history"}).get("chat_history")}
        | query_rewrite_prompt
        | llm
        | StrOutputParser()
        )
        print("Function: get_query_rewrite_chain \t Status: Completed")
        return query_rewrite_chain
    def get_query_version_extractor_chain(self,version):
        print("Function: get_query_version_extractor_chain \t Status: Started")
        def get_version(_):
          return version
        query_rewrite_vl_prompt = PromptTemplate(template=query_rewrite_versionless_template_3, input_variables=['query','target'])
        query_rewrite_versionless_chain = ({
        "query": RunnablePassthrough(),
        "target": RunnableLambda(get_version)}
        | query_rewrite_vl_prompt
        | self.llm
        | StrOutputParser()
        )
        print("Function: get_query_version_extractor_chain \t Status: Completed")
        return query_rewrite_versionless_chain

    def rewrite_query(self, query):
        '''invoke the rewriter chain and format the outputs'''
        print("Function: Query Rewriting Pipeline \t Status: Started")
        free_gpu_memory()
        try:
            response = self.query_rewrite_chain.invoke(query,config={'callbacks': [StdOutCallbackHandler()]})
            # print("Rewritten Query:\t",response)
            standalone_question,version,better_query = self.format_rewrite_answer(query, response)
            queries = []
            ch = self.memory.load_memory_variables({"chat_history"}).get("chat_history") or 'None'
            standalone_question = standalone_question if ch != 'None' else query
            queries.append(standalone_question)
            queries.append(better_query)
            bc = None
            try:
                if version != 'None':
                    def get_available_versions(_):
                        names = [item["name"] for item in documen_corpus_type]
                        names.append("None")
                        formatted_string = "\n".join([f"{index + 1}. {name}" for index, name in enumerate(names)])
                        return formatted_string
                    v_chain = ({"list_of_versions":RunnableLambda(get_available_versions), "target":RunnablePassthrough()} 
                    | PromptTemplate(
                    template = check_version_prompt_llama3,
                    input_variables = ["list_of_versions", "target"]) 
                    | self.llm 
                    | StrOutputParser()
                    )
                    res = v_chain.invoke(version)
                    import re
                    fd = re.sub(r'\\([^n]|$)', '', res)
                    res_parsed=parse_json_garbage('{'+fd)['index']
                    fv = 'None' if len(documen_corpus_type) < int(res_parsed) else documen_corpus_type[int(res_parsed) - 1]['name']
                    if fv != 'None':
                        print(version)
                        query_rewrite_versionless_chain = self.get_query_version_extractor_chain(version)
                        versonless = query_rewrite_versionless_chain.invoke(better_query,config={'callbacks': [StdOutCallbackHandler()]})
                        vl = parse_json_garbage(versonless)
                        queries.append(vl['answer'])
                        queries.append(vl['complete_answer'])
                        bc = parse_json_garbage(versonless)['complete_answer']
                        version = fv
            except Exception as e:
                print(f"Error: {e}")
                return [standalone_question,better_query], standalone_question, 'None', query
            queries = list(set(queries))
            print("Function: Query Rewriting Pipeline \t Status: Completed")
            return queries, standalone_question, version, bc
        except Exception as e:
            print(f"Error: {e}")
            return [query], query, 'None', query

    
    def format_rewrite_answer(self,question, rewritten_query):
        try:
            json_obj = parse_json_garbage(rewritten_query)
            standalone_question = json_obj["standalone_question"]
            return standalone_question, json_obj['version'],json_obj['better_query']
        except Exception as E:
            print(f"Error occured while formatting the LLM query rewriting answer.\n Query: {question}\n LLM Answer: {rewritten_query}\n Exception: {E}")

    
