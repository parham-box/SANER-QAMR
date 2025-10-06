import os
from torch import bfloat16
import pickle
import uuid
from constants.config import *
from constants.prompts import *
from constants.data import *
from constants.helpers import *
from constants.retriever_type import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
import tiktoken
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
import fitz
import re
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline


#Global variable
enc = tiktoken.get_encoding("cl100k_base")

def save_to_pickle(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def load_from_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

def get_multi_vector_retriever(doc_splits, extended_splits:None, k, chunk_size, chunk_overlap, is_all_pages=False, chunks=None,sl=None):
  print("Function: get_multi_vector_retriever \t Status: Started") 
  embed_model_kwargs = {"device":device}
  bge_embed_model_id = "BAAI/bge-base-en-v1.5"
  bge_embed_encode_kwargs = {'device': device, 'normalize_embeddings': True}
  # create a hugging face embedding to be used to convert the document splits to embedings so they can be added to the db
  bge_embed = HuggingFaceEmbeddings(
      model_name=bge_embed_model_id,
      model_kwargs=embed_model_kwargs,
      encode_kwargs=bge_embed_encode_kwargs
  )
  pd = persist_directory
  os.makedirs(pd, exist_ok=True)
  if sl != None:
    pd = os.path.join(pd, sl)
    os.makedirs(pd, exist_ok=True)
  vector_store_path = os.path.join(pd, "chroma/")
  docstore_path = os.path.join(pd, "docstore.pkl")

  vectorstore = Chroma(
      collection_name="full_documents", embedding_function=bge_embed, persist_directory=vector_store_path
  )
  # The storage layer for the parent documents
  store = InMemoryByteStore()
  id_key = "doc_id"
  # The retriever (empty to start)
  retriever1 = MultiVectorRetriever(
      vectorstore=vectorstore,
      byte_store=store,
      id_key=id_key,
      search_kwargs={"k": k}
  )  
  retriever2 = MultiVectorRetriever(
      vectorstore=vectorstore,
      byte_store=store,
      id_key=id_key,
      search_type="mmr", 
      search_kwargs={"k": k, "include_metadata":True}
  )
  doc_ids = [str(uuid.uuid4()) for _ in doc_splits]
  sub_docs = []
  for i, doc in enumerate(doc_splits):
      if chunks== None:
        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
      else:
        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunks[doc.metadata['source']], chunk_overlap=chunks[doc.metadata['source']]/4)
      _id = doc_ids[i]
      if is_all_pages == True:
        doc.metadata[id_key] = _id
      else:
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        sub_docs.extend(_sub_docs)
  if is_all_pages == True:
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    for batch in chunks(doc_splits, 41600):
      retriever1.vectorstore.add_documents(batch)
      print(f"Processed batch of size {len(batch)}")
  else:
    # Function to chunk the list into batches
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    for batch in chunks(sub_docs, 41600):
        retriever1.vectorstore.add_documents(batch)
        print(f"Processed batch of size {len(batch)}")
  retriever1.docstore.mset(list(zip(doc_ids, extended_splits)))

  from langchain.retrievers import MergerRetriever
  lotr = MergerRetriever(retrievers=[retriever1, retriever2])


  print("Function: load_multi_vector_retriever \t Status: Completed")  
  reordering = LongContextReorder()
  pipeline = DocumentCompressorPipeline(transformers=[reordering])
  # use contextual compression to create the retriever: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/
  reordered_retriever = ContextualCompressionRetriever(
      base_compressor=pipeline, base_retriever=lotr
  )

  save_to_pickle(retriever1.byte_store.store, docstore_path)

  print("Function: get_multi_vector_retriever \t Status: Completed")  
  return reordered_retriever
def load_multi_vector_retriever(k,sl=None,dt=None):
  print("Function: load_multi_vector_retriever \t Status: Started") 
  embed_model_kwargs = {"device":device}
  bge_embed_model_id = "BAAI/bge-base-en-v1.5"
  bge_embed_encode_kwargs = {'device': device, 'normalize_embeddings': True}
  # create a hugging face embedding to be used to convert the document splits to embedings so they can be added to the db
  bge_embed = HuggingFaceEmbeddings(
      model_name=bge_embed_model_id,
      model_kwargs=embed_model_kwargs,
      encode_kwargs=bge_embed_encode_kwargs
  )
  pd = persist_directory
  os.makedirs(pd, exist_ok=True)
  if sl != None:
    pd = os.path.join(pd, sl)
  vector_store_path = os.path.join(pd, "chroma/")
  docstore_path = os.path.join(pd, "docstore.pkl")

  vectorstore = Chroma(
      collection_name="full_documents", embedding_function=bge_embed, persist_directory=vector_store_path
  )
  store_dict = load_from_pickle(docstore_path)
  store = InMemoryByteStore()
  store.mset(list(store_dict.items()))
  # The storage layer for the parent documents
  id_key = "doc_id"
  # The retriever (load from file)
  retriever1 = MultiVectorRetriever(
      vectorstore=vectorstore,
      byte_store=store,
      id_key=id_key,
      search_kwargs={"k": k}
  )  
  retriever2 = MultiVectorRetriever(
      vectorstore=vectorstore,
      byte_store=store,
      id_key=id_key,
      search_type="mmr", 
      search_kwargs={"k": k, "include_metadata":True}
  )
  from langchain.retrievers import MergerRetriever
  lotr = MergerRetriever(retrievers=[retriever1, retriever2])


  print("Function: load_multi_vector_retriever \t Status: Completed")  
  reordering = LongContextReorder()
  pipeline = DocumentCompressorPipeline(transformers=[reordering])
  # use contextual compression to create the retriever: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/
  reordered_retriever = ContextualCompressionRetriever(
      base_compressor=pipeline, base_retriever=lotr
  )
  return {"name": dt['name'], "retriever": reordered_retriever, "default":dt['default']}
def make_text(words):
    """Return textstring output of get_text("words").

    Word items are sorted for reading sequence left to right,
    top to bottom.
    """
    line_dict = {}  # key: vertical coordinate, value: list of words
    words.sort(key=lambda w: w[0])  # sort by horizontal coordinate
    for w in words:  # fill the line dictionary
        y1 = round(w[3], 1)  # bottom of a word: don't be too picky!
        word = w[4]  # the text of the word
        line = line_dict.get(y1, [])  # read current line content
        line.append(word)  # append new word
        line_dict[y1] = line  # write back to dict
    lines = list(line_dict.items())
    lines.sort()  # sort vertically
    lin = [" ".join(line[1]) for line in lines]
    l = [item for item in lin if "Copyright©" not in item]
    return "__\n".join(l)

def load_single_document(source_location,all_pages):
  '''Load the single document'''
  print("Function: load_single_document \t Status: Started")
  # load_single_document_with_table_inference(source_location, model=get_sum_model())
  #FOR TEXT LOADING
  base_name = os.path.basename(source_location)
  file_extension = os.path.splitext(base_name)[1].lower()
  output = []
  if file_extension == '.txt':
      loader = TextLoader(source_location, encoding="utf-8")
      documents = loader.load()
      print("Function: load_single_document \t Status: Completed")
      return documents
  elif file_extension == ".pdf":
    d = fitz.open(source_location)
    title = d.metadata['title']
    txt_filename = os.path.splitext(base_name)[0] + ".txt"
    txt_file_path = os.path.join("test_text_documents", txt_filename)
    # Ensure the directory exists
    os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
    with open(txt_file_path, "w",encoding="utf-8") as file:
        document_size = len(d)
        for page_num,page in enumerate(d.pages()):
          words = page.get_text("blocks")
          mywords = [w for w in words if fitz.Rect(w[:4])] #not read header or footer
          content = make_text(mywords) #make the words in the correct order
          patterns = [r'(•)\n', r'(–)\n', r'(—)\n'] #format the bullet points and dashes
          # Replace each pattern with just the bullet point or en dash
          for pattern in patterns:
            content = re.sub(pattern, r'\1', content)
          content = re.sub(r'[\r\n]+', '\n', content) #remove extra new line chars
          # content = re.sub(r' +', ' ', content)
          # content = "from: " + title + '\n' + content
          file.write(content + '\n')
          output.append(Document(page_content=content, metadata={"source": base_name, "page": page_num}))
    print("Function: load_single_document \t Status: Completed")
    if all_pages:
      loader = TextLoader(txt_file_path, encoding="utf-8")
      documents = loader.load()
      return documents
    else:
      return output
  elif file_extension == '.docx':
     from langchain_community.document_loaders import Docx2txtLoader
     txt_filename = os.path.splitext(base_name)[0] + ".txt"
     loader = Docx2txtLoader(source_location)
     documents = loader.load()
     content = documents[0].page_content
     metadata = documents[0].metadata
     content = re.sub(r'[\r\n]+', '\n', content) #remove extra new line chars
     content = re.sub(r'\t+', '', content)
     print("Function: load_single_document \t Status: Completed")
     return [Document(page_content=content, metadata=metadata)]

def add_context_to_pages(doc_splits, chunk_overlap):
    doc_extended = []
    av = 0
    ind = 1
    for i, doc in enumerate(doc_splits):
        # Extract content of the current page
        page_content = doc.page_content
        av += len(page_content)
        # Extract content of the previous page if it exists and has at least 500 characters
        prev_page_content = doc_splits[i - 1].page_content[-chunk_overlap:] if i > 0 and len(doc_splits[i - 1].page_content) >= chunk_overlap else ""
        
        # Extract content of the next page if it exists and has at least 500 characters
        next_page_content = doc_splits[i + 1].page_content[:chunk_overlap] if i < len(doc_splits) - 1 and len(doc_splits[i + 1].page_content) >= chunk_overlap else ""
        
        # Combine content of the current, previous, and next pages
        extended_page_content = prev_page_content + '\n' + page_content + '\n' + next_page_content
        # Create a new Document object with the extended content
        extended_doc = Document(page_content=extended_page_content, metadata=doc.metadata)
        ind += 1
        # Add the extended document to the list
        doc_extended.append(extended_doc)
    
    return doc_extended, av/ind

def load_documents(source_location, all_pages=False):
  '''Load all documents'''

  print("Function: load_documents \t Status: Started")
    # Recursively find all file paths in the source_location
  file_paths = []
  for root, dirs, files in os.walk(source_location):
      for file in files:
          file_paths.append(os.path.join(root, file))
  print(len(file_paths))
  docs = []
  with ProcessPoolExecutor() as executor:
      futures = []
      # Process in batches of 50
      for i in range(0, len(file_paths), 50):
          batch = file_paths[i:i + 50]
          for file_path in batch:
              future = executor.submit(load_single_document, file_path, all_pages)
              futures.append(future)
          # Collect the results as the futures complete
          for future in as_completed(futures):
              content = future.result()
              docs.append(content)
          # Clear the futures list after processing each batch
          futures.clear()

  print("Function: load_documents \t Status: Completed")
  return docs


def token_length_function(text: str) -> int:
    return len(enc.encode(text))

def split_documents(document, chunk_size,chunk_overlap):
  '''Split the documents and return the chunks'''

  print("Function: split_documents \t Status: Started")
  # Keeping the chunk_overlap at 25% of the chunk_size
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  splits = text_splitter.split_documents(document)

  print("Function: split_documents \t Status: Completed")
  return splits

def get_retrievers(type, chunk_size=0, child_chunk_size=0, chunk_overlap=0,child_chunk_overlap=0,k=3, extra_context_overlap=0):
  '''Wrapper function for the retrieval process'''
  print("Function: get_retriever \t Status: Started")  
  free_gpu_memory()
    #Load the documents
  if load_presist == 'n':
    os.system(f'rm -rf {persist_directory}')
    all_pages = False
    retrievers = []
    print("TYPE",type)
    for dt in documen_corpus_type:
      source_location = dt['document_location']
      source_version = dt['name']
      documents = load_documents(source_location,all_pages)
      doc_exteneded_splits=[]
      doc_splits=[]
      retriever=None
      if type == RetrieverType.EXTRA_CONTENT_PAGE_CHUNK_BY_MEDIAN:
        print("Function: Get RetrieverType.EXTRA_CONTENT_PAGE_CHUNK_BY_MEDIAN \t Status: Started")  
        chunks = {}
        for document in documents:
          ex,av = add_context_to_pages(doc_splits=document, chunk_overlap=extra_context_overlap)
          rr = round(av,-2)
          chunks[document[0].metadata['source']] = rr / 2 if rr > 1000 else rr
          doc_exteneded_splits.extend(ex)
          doc_splits.extend(document)
        free_gpu_memory()
        retriever = get_multi_vector_retriever(doc_splits=doc_splits,extended_splits=doc_exteneded_splits,k=k,chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap, is_all_pages=False, chunks=chunks,sl=source_location)
        print("Function: Get RetrieverType.EXTRA_CONTENT_PAGE_CHUNK_BY_MEDIAN \t Status: Completed")  

      else:
          return_value = "Invalid type"
      reordering = LongContextReorder()
      pipeline = DocumentCompressorPipeline(transformers=[reordering])
      reordered_retriever = ContextualCompressionRetriever(
          base_compressor=pipeline, base_retriever=retriever
      )
      retrievers.append({"name": source_version, "retriever": reordered_retriever, "default":dt['default']})
         
    print("Function: get_retriever \t Status: Completed")  
    return retrievers
  else:
    print("Function: get_retriever \t Status: Completed")  
    return [load_multi_vector_retriever(k=2,sl=dt['document_location'],dt=dt) for dt in documen_corpus_type]