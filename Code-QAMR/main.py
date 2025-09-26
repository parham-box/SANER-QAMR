
from constants.config import *
from constants.prompts import *
from constants.data import *
from evaluation import *
from constants.helpers import *
from constants.retriever_type import *
from retrievers import *
from retrieval_augmented_generation import *
from rewriter import Rewriter
import pandas as pd

def main():
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

  answers = []
  context = []
  metadata = []
  times = []

  df, columns = xlsx_columns_to_lists(f'REQuestA-Final-srs.xlsx')
  questions = columns['Question']
  gt = columns['Text Passage']
  gta = columns['Answer']


  import time
  with open('results_requesta/docs.txt', 'w') as file:
    pass
  for index,question in enumerate(questions):
      free_gpu_memory()
      start_time = time.time()  # Start time measurement
      queries,sa,version,bc = query_rewriter.rewrite_query(query= question)
      answer = rag.answer_question_multiple_query(query=queries, sa=sa,bc=bc,version=version,file_name=file_name)
      end_time = time.time()  # Start time measurement
      with open('results_requesta/docs.txt', 'a') as file:
        file.write(f"Answer generated in: {end_time - start_time}")
      answers.append(answer['answer'])
      context.append([docs.page_content for docs in answer['context']])
      metadata.append([docs.metadata for docs in answer['context']])
      times.append(end_time - start_time)
  df['Generated_Answer'] = answers
  df['Found_Contexts'] = context
  df['Found_Metadata'] = metadata
  df['times'] = times

  # Write the updated DataFrame back to the same XLSX file
  df.to_excel(f'{loc}Answered-REQuestA-Final-srs.xlsx', index=False)
  print("Function: main \t Status: Completed")

if __name__ == "__main__":
    main()





