from constants.config import *
from constants.prompts import *
from constants.data import *
from constants.helpers import *
from constants.retriever_type import *
   
class deepeval_evaluation():
   def __init__(self,answers, contexts,questions, gt,loc):
    from deepeval_eval import CustomLlama3_70B
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.dataset import EvaluationDataset
    from deepeval import evaluate
    from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric, 
    FaithfulnessMetric,
    GEval
)
    custom_llm = CustomLlama3_70B()
    contextual_precision = ContextualPrecisionMetric(model=custom_llm)
    contextual_recall = ContextualRecallMetric(model=custom_llm)
    contextual_relevancy = ContextualRelevancyMetric(model=custom_llm)
    answer_relevancy_metric = AnswerRelevancyMetric(model=custom_llm)
    faithfulness = FaithfulnessMetric(model=custom_llm)
    correctness_metric = GEval(
        name="Correctness",
        model=custom_llm,
        evaluation_params=[
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Compare the 'actual output' directly with the 'expected output' to verify factual accuracy",
            "Any mismatch in units or precision values is NOT acceptable and must result in score of zero",
            "Assess if there are any discrepancies in details, values, or information between the actual and expected outputs",
            "It is OK if the 'actual output' omits some details from 'expected output' as long as the 'actual output' answers the input",
            "It is OK for the 'actual output' to include extra information or present details in a different order than the 'expected output'",
            "Missing examples are acceptable as long as the main concepts between the 'expected output' and 'actual output' are the same",
            "If the 'expected output' is not empty and does not explicitly state that the input cannot be answered, then 'actual output' of 'I don't know' is NOT acceptable and must result in a score of zero",
            "If the 'expected output' is empty or explicitly states that the input cannot be answered, then the only acceptable 'actual output' is 'I don't know,' and it should receive full credit" 
     ]
    )
    test_cases = []
    for answer, context, question, ground_truth in zip(answers, contexts, questions, gt):
        import ast
        actual_list = ast.literal_eval(str(context))

        test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=ground_truth,
        retrieval_context=actual_list,
        )
        test_cases.append(test_case)

    dataset = EvaluationDataset(test_cases=test_cases)
    ev = evaluate(dataset, [contextual_precision,contextual_recall,contextual_relevancy,answer_relevancy_metric,faithfulness,correctness_metric], ignore_errors=True, print_results=True,verbose_mode=True)
    with open(f"{loc}deepeval.txt", "a") as file:
       file.write(f"{ev}\n\n")
