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
from operator import itemgetter
import re
import time
from langchain.globals import set_verbose
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.documents import Document
from langchain.retrievers.multi_query import _unique_documents
set_verbose(True)

class RAG():
    def __init__(self,retriever,memory,r2=None):
        self.memory = memory
        '''Create llm pipeline with the model'''
        llm_pipeline = get_llm_pipeline(get_model(model_path,model_id), model_path)
        self.llm = llm_pipeline
        self.r2 = r2
        '''Create the rag chain'''
        self.rag_chain = self.get_rag_chain(llm_pipeline,retriever)
    def get_pipeline(self):
        return self.llm
    def retrieve_docs_for_questions(self,retriever,questions,sa,version,bc):
        start_time_release_mapping = time.time()  # Start time measurement

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
        if version != 'None':
            # fv = version
            res = v_chain.invoke(version)
            import re
            fd = re.sub(r'\\([^n]|$)', '', res)
            res_parsed=parse_json_garbage('{'+fd)['index']
            fv = 'None' if len(documen_corpus_type) < int(res_parsed) else documen_corpus_type[int(res_parsed) - 1]['name']
        else:
            fv = 'None'
        matching_retrievers = []
        ds = ""
        for entry in retriever:
            if fv == 'None':
                if entry['default'] == 'Yes':
                    ds = entry['name']
                    matching_retrievers.append(entry['retriever'])
            else:
                if entry['name'] == fv:
                    ds = entry['name']
                    matching_retrievers.append(entry['retriever'])

        if not matching_retrievers:
            for entry in retriever:
                if entry['default'] == 'Yes':
                    ds = entry['name']
                    matching_retrievers.append(entry['retriever'])
        if not matching_retrievers:
                raise ValueError(f"No retrievers found for version '{fv}'")

        print("Question asked from documents located in: ", ds)
        reordered_retriever = matching_retrievers[0]
        end_time_release_mapping = time.time()  # Start time measurement
        release_mapping_times.append(end_time_release_mapping - start_time_release_mapping)

        fc_chain11 = ({"contexts":itemgetter("contexts"), "question":itemgetter("question"), "version":itemgetter("version")} 
        | PromptTemplate(
        template = question_answer_extraction_llama3,
        input_variables = ["contexts", "question","version"]) 
        | self.llm 
        | StrOutputParser()
        )
        start_time_document_retrieval = time.time()
        all_docs = []
        import hashlib
        def generate_doc_hash(doc):
            # Create a hash based on the document content
            return hashlib.sha256(doc.encode()).hexdigest()

        def create_unique_id(metadata,content):
            page = metadata.get('page', generate_doc_hash(content))  # Use 'unknown' if 'page' key is not present
            return f"{metadata['source']}_{page}"

        for question in questions:
            docs = reordered_retriever.get_relevant_documents(question)
            all_docs.extend([{'doc':doc, 'q':question}for doc in docs])
        unique_docs = []
        seen_ids = set()
        for item in all_docs:
            doc_id = create_unique_id(item['doc'].metadata,item['doc'].page_content)
            if doc_id not in seen_ids:
                unique_docs.append(item)
                seen_ids.add(doc_id)  
        end_time_document_retrieval = time.time()  # Start time measurement
        document_retrieval_times.append(end_time_document_retrieval - start_time_document_retrieval)
  
        with open('results_requesta/docs.txt', 'a') as file:
            file.write(f"Before Answer Formation:\n")
            for d in unique_docs:
                file.write(f"{d['doc'].metadata}\n")
            file.write(f"\n")

            df = self.answer_reranking(unique_docs,version,sa,fc_chain11)

            file.write(f"After Answer Formation:\n")
            for d in df:
                file.write(f"{d['doc'].metadata}\n")
            
            file.write(f"-----\n")
        # return [dff[id]['doc'] for id in answers if id < len(dff)]
        # return [doc['doc'] for doc in dff]
        return [doc['doc'] for doc in df]

    def get_rag_chain(self,llm,retriever):
        # create RAG pipeline. it uses the retirever to get 3 documents and passes them to the prompt as context together with the question so the llm can write the final response
        rag_pipeline_d = (    
        RunnablePassthrough.assign(context=(lambda x: self.format_docs_answer_extraction(x["context"])))
        | PromptTemplate(
            template = question_answering_answer_extraction_prompt_llama3_public,
            # template = get_prompt(type="QA"),
            input_variables = ["version","context", "question"]
        ) 
        | print_prompt
        | llm
        | StrOutputParser()
        )  
        retrieve_docs = (lambda x: self.retrieve_docs_for_questions(retriever, x["questions"],x['question'], x['version'],x['bc']))
        from langchain_core.runnables import RunnableBranch
        branch = RunnableBranch(
            (lambda x: not x["context"], RunnablePassthrough.assign(answer=lambda x: "I don't know")),
            rag_pipeline_d
        )

        # Create the full RAG pipeline
        rag_pipeline = (
            RunnablePassthrough.assign(context=retrieve_docs)
            .assign(answer=branch)
            .assign(answer=lambda x: x["answer"]["answer"] if isinstance(x["answer"], dict) and "answer" in x["answer"] else x["answer"])
            .assign(answer= lambda x: x['answer'] if x['answer'] != '' else "I don't know")
        )
        return rag_pipeline

    def answer_question_multiple_query(self, query,sa,bc,version,file_name):
        '''invoke the rag chain to get answer, and add it to memory'''
        free_gpu_memory()
        answer = self.rag_chain.invoke({"questions":query, "question": sa, "version": version,"bc":bc} , config={'callbacks': [StdOutCallbackHandler()]})  
        # print("ANSSSS",answer)
        final_answer = self.write_answer_and_sources_to_file(file_name,answer['questions'],answer,version)
        self.memory.save_context({"question": answer['question']},{"output": final_answer['answer']})
        return final_answer
    
    def format_docs_answer_extraction(self, docs, char_limit=15000):
        '''Format retrieved chunks and adding them together'''
        char_count = 0
        combined_content = []
        for i,doc in enumerate(docs):
            if char_count + len(doc.page_content) <= char_limit:
                combined_content.append(f"Potential Answer {i+1}:\n{doc.page_content}")
                char_count += len(doc.page_content)
            else:
                break
        return "\n__\n".join(combined_content)
    def answer_reranking(self, unique_docs, version, sa,fc_chain11):
        start_time_context_selection = time.time()
        def format_contexts(contexts):
            formatted_contexts = ""
            for i, context in enumerate(contexts):
                formatted_contexts += f"context{i+1}:\n {context.page_content}\n"
            return formatted_contexts
        with open('results_requesta/docs.txt', 'a') as file:
            cts = format_contexts([doc['doc'] for doc in unique_docs])
            # dfff = []
            asd = fc_chain11.invoke({"contexts":cts, "version": version, "question": sa})
            asd = asd.replace("'", "\"")
            try:
                asdd= parse_json_garbage('{'+ asd )
                answers1 = [int(id) -1 for id in asdd['ids']]
            except Exception as e:
                answers1 = [id for id in range(len(cts))]
            file.write(f"AFTER EXTRACTION: {answers1}\n")
        
        first_three_ids = answers1[:3]   
        dff = [unique_docs[id] for id in first_three_ids if id >= 0  and id < len(unique_docs)]  
        end_time_context_selection = time.time()
        context_selection_times.append(end_time_context_selection - start_time_context_selection)     
        return dff    

    def write_answer_and_sources_to_file(self,file_name,question,answer,version=None):
        '''Extract and write the answer and sources to the result file'''
        with open(file_name, "a",encoding="utf-8") as file:
            file.write(f"{question}\n")
            if version:
                file.write(f"{version}\n")

            # print(answer)
            # json_obj = parse_json_garbage('{'+answer["answer"])

            file.write("The answer was generated base on the following:\n")
            for document in answer['context']:
                name = document.metadata["source"]
                page = document.metadata.get("page")
                if page is not None:
                    file.write(f"{name} - page: {page+1}\n")
                else:
                    file.write(f"{name}\n")
            file.write(f"Answer:\t{answer['answer']}\n------\n")
        return answer
