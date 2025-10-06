
from constants.config import *
from constants.prompts import *
from constants.data import *
from evaluation import *
from constants.helpers import *
from constants.retriever_type import *
from retrievers import *
from rewriter import Rewriter
import pandas as pd

def main_evaluation():
  
  df, columns = xlsx_columns_to_lists(f'{loc}Answered-REQuestA-Final-srs.xlsx')
  questions = columns['Question']
  gt = columns['Text Passage']
  gta = columns['Answer']
  answers = columns['Generated_Answer']
  context = columns['Found_Contexts']


  dep = deepeval_evaluation(answers=answers, contexts=context,questions=questions, gt=gta,loc=loc)
  extract_evaluation(loc,'deepeval.txt')

if __name__ == "__main__":
    main_evaluation()





