import gc
import torch
import json
from GPUtil import showUtilization as gpu_usage
from langchain.memory import ConversationBufferWindowMemory
import transformers
from constants.config import *
from constants.prompts import *
from constants.data import *
from langchain_huggingface import HuggingFacePipeline
from langchain.globals import set_verbose
set_verbose(True)

def print_prompt(prompt):
    print("\033[92m" + prompt.text + "\033[0m")
    return prompt

def get_model(path, id):
  '''Load the LLM'''

  print("Function: get_model \t Status: Started")
  model = transformers.AutoModelForCausalLM.from_pretrained(
      path,
      quantization_config=bnb_config,
      device_map=device,
      token=hf_auth,
  )
  model.eval()
  print(f"Model: {id} loaded on {device}")

  print("Function: get_model \t Status: Completed")
  return model

def get_llm_pipeline(model, path):
  '''Create the transformer LLM pipeline with the loaded model'''

  print("Function: get_llm_pipeline \t Status: Started")

  # Loading the Llama2 tokenizer
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      path,
      token=hf_auth
  )

  # Transformer pipeline parameters
  #create a pipeline for abstraction: https://huggingface.co/docs/transformers/en/main_classes/pipelines
  transformer_pipeline = transformers.pipeline(
      model=model,
      tokenizer=tokenizer,
      return_full_text=False,
      task='text-generation',
      temperature=0.01,
      max_length=50000,
      repetition_penalty=1.15,
      pad_token_id= tokenizer.eos_token_id
  )

  # Creating the LLM pipeline
  llm = HuggingFacePipeline(pipeline= transformer_pipeline)

  print("Function: get_llm_pipeline \t Status: Completed")
  return llm

def get_memory():
  '''Initialize the conversation memory'''

  print("Function: get_memory \t Status: Started")

  memory = ConversationBufferWindowMemory(k=0, input_key="question", memory_key="chat_history")

  print("Function: get_memory \t Status: Completed")
  return memory

def get_prompt(type):
  '''Return the prompt template'''
  if type == "QA":
    print("Function: get_prompt \t Status: Started")
    template = question_answering_answer_extraction_prompt_llama3_public
    print("Function: get_prompt \t Status: Completed")
  elif type == "Rewrite":
    print("Function: get_rewrite_prompt \t Status: Started")
    template = query_rewrite_template_3
    print("Function: get_rewrite_prompt \t Status: Completed")
  return template

def parse_json_garbage(s):
    import re
    s = re.sub(r'```json', '', s)
    s = re.sub(r'```', '', s)
    s = s.replace("\\'", "'").replace("\\\\", "\\")
    s = s[next(idx for idx, c in enumerate(s) if c in "{"):]
    try:
        return json.loads(s, strict=False)
    except json.JSONDecodeError:
        try:
            return json.loads(s + "}")
        except json.JSONDecodeError as e:
             print(ValueError(f"Unable to parse JSON string.{e}"))
             return 'None'
        
def free_gpu_memory():
    '''Free up GPU memory'''

    print("GPU usage before releasing memory")
    gpu_usage()
    torch.cuda.empty_cache()

    gc.collect()

    print("GPU Usage after releasing memory")
    gpu_usage()
    
def extract_evaluation(loc,path):
  print("Function: extract evaluation \t Status: Started")
  import re
  with open(f"{loc}{path}", 'r') as file:
    content = file.read()

  # Regular expression to match MetricData names and scores
  pattern = r'MetricData\(name=["\'](.*?)["\'].*?score=(\d+\.\d+).*?reason=["\'](.*?)["\'].*?evaluation_model=["\'](.*?)["\'].*?error=(None|["\'](.*?)["\'])'
  # Find all matches in the content
  matches = re.findall(pattern, content)

  results = []
  faithfulness_results = []
  answer_relevancy_results = []
  context_percision_results = []
  context_recall_results = []
  context_relevancy_results = []
  answer_correctness = []
  # Extract and print the metric names and scores
  for match in matches:
      metric_name, score, reason, evaluation_model, error, error_str = match
      # Determine the correct error value
      error = error if error == None else error_str
      obj = {
          "metric": metric_name,
          "score": score,
          "evaluation_model": evaluation_model,
          "reason": reason,
          "error": error
      }
      results.append(obj)
      if metric_name == "Faithfulness":
          faithfulness_results.append(obj)
      elif metric_name == "Answer Relevancy":
          answer_relevancy_results.append(obj)
      elif metric_name == "Contextual Precision":
          context_percision_results.append(obj)
      elif metric_name == "Contextual Recall":
          context_recall_results.append(obj)
      elif metric_name == "Contextual Relevancy":
          context_relevancy_results.append(obj)
      elif metric_name == "Correctness (GEval)":
          answer_correctness.append(obj)

  import pandas as pd
  df_all = pd.DataFrame(results)
  df_f = pd.DataFrame(faithfulness_results)
  df_ar = pd.DataFrame(answer_relevancy_results)
  df_crel = pd.DataFrame(context_relevancy_results)
  df_cp = pd.DataFrame(context_percision_results)
  df_crec = pd.DataFrame(context_recall_results)
  df_ac = pd.DataFrame(answer_correctness)
  df_all.to_csv(f"{loc}all_res_public.csv", index=False)
  df_f.to_csv(f"{loc}f_res_public.csv", index=False)
  df_ar.to_csv(f"{loc}arel_res_public.csv", index=False)
  df_crel.to_csv(f"{loc}crel_res_public.csv", index=False)
  df_cp.to_csv(f"{loc}cp_res_public.csv", index=False)
  df_crec.to_csv(f"{loc}crec_res_public.csv", index=False)
  df_ac.to_csv(f"{loc}ac_res_public.csv", index=False)
  print("Function: extract evaluation \t Status: Completed")

def xlsx_columns_to_lists(file_path):
    import pandas as pd
    '''Reads an XLSX file and returns a dictionary with column names as keys and column contents as lists'''
    df = pd.read_excel(file_path)  # Read the Excel file into a DataFrame
    columns_dict = df.to_dict(orient='list')  # Convert DataFrame to dictionary of lists
    return df, columns_dict
