
from constants.config import *
from constants.prompts import *
from constants.data import *
from evaluation import *
from constants.helpers import *
from constants.retriever_type import *
from retrievers import *
from retrieval_augmented_generation import *
from rewriter import Rewriter
import pickle
import pandas as pd

def main_time():
  print("Function: main \t Status: Started")
  free_gpu_memory()

  '''Get Memory'''
  memory = get_memory()

  '''Define the type and parameters of the retriever'''
  ret_type = RetrieverType.EXTRA_CONTENT_PAGE_CHUNK_BY_MEDIAN
  parmaters = (0,0,0,0,500,2)

  '''Get the retriever, split the documents and embed the data in the vectorstore'''
  retriever = get_retrievers(
  type=ret_type,
  chunk_size=parmaters[0],
  child_chunk_size=parmaters[2],
  chunk_overlap=parmaters[1],
  child_chunk_overlap=parmaters[3],
  k=parmaters[5],
  extra_context_overlap=parmaters[4]
  )

  '''Get RAG chain'''
  rag = RAG(retriever,memory,None)
  llm = rag.get_pipeline()

  '''Get Query Rewriter chain'''
  query_rewriter = Rewriter(memory,llm)
  
  # Create the result file name
  param_string = '_'.join(map(str, parmaters))
  file_name = os.path.join(results_directory,f"{ret_type}_{param_string}.txt")
  print(file_name)
  with open(file_name, "w") as file:
    pass  # Empty the file  

  answers = []
  context = []
  metadata = []
  times = []

  df, columns = xlsx_columns_to_lists(f'REQuestA-Final-srs.xlsx')
  questions = columns['Question']
  gt = columns['Answer']
  question_gt_pairs = list(zip(questions, gt))
  import random
  # Sample 40 random question-gt pairs
  sampled_pairs = random.sample(question_gt_pairs, 40)
  questions = [pair[0] for pair in sampled_pairs]
  gt = [pair[1] for pair in sampled_pairs]

  import time
  with open('results_requesta/docs.txt', 'w') as file:
    pass
  for index,question in enumerate(questions):
      free_gpu_memory()
      start_time = time.time()  # Start time measurement
      queries,sa,version,bc = query_rewriter.rewrite_query(query= question)
      end_time_query_rewriting = time.time()  # Start time measurement
      query_rewting_times.append(end_time_query_rewriting - start_time)
      answer = rag.answer_question_multiple_query(query=queries, sa=sa,bc=bc,version=version,file_name=file_name)
      end_time = time.time()  # Start time measurement
      with open('results/docs.txt', 'a') as file:
        file.write(f"Answer generated in: {end_time - start_time}")
      answers.append(answer['answer'])
      context.append([docs.page_content for docs in answer['context']])
      metadata.append([docs.metadata for docs in answer['context']])
      times.append(end_time - start_time)
  result_df = pd.DataFrame({
    'Sampled_Questions': questions,
    'Ground_Truth': gt,
    'Generated_Answer': answers,
    'Found_Contexts': context,
    'Found_Metadata': metadata,
    'Times': times,
    'Query_Rewriting_Times': query_rewting_times,
    'Release_Mapping_Times': release_mapping_times,
    'Document_Retrieval_Times': document_retrieval_times,
    'Context_Reduction_Times': context_reduction_times,
    'Context_Selection_Times': context_selection_times,
  })
  result_df.to_excel(f"Answered-QAMR-Timed.xlsx", index=False)
  print("Function: main \t Status: Completed")

if __name__ == "__main__":
    main_time()





