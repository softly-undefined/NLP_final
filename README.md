# NLP_Final Project (we should think of a name for the paper!)
Boring: Cognitive Alignment for MCQ Answering with LLMs.

## What's next
1. generate the rest of the data

after the current batch finishes (about 40 hours) still have to do approach_2 for both english and chinese and then done!

2. analyze the data

Get all the metrics from the data to analyze : )
(how to do confidence? we should discuss) (maybe second round of llm prompting for easyness?)

i have theoretically solved the issue of not having confidence values i think! using our embedding values we can theoretically meausre how "easy" a question is. We can then take a subset of very 'easy' questions and measure the order bias of those questions to see if it is the same or different

maybe not the best 


## 0_Starter Code:
This code isn't actually used in the project, it just contains reference code, both written by us and from other sources (primarily OpenAI's simpleeval). We will probably remove this directory in the final step of the project.

## 0_First Test:
This code also isn't used in the project! It was our first wider scale tests with categorization and evaluation before realizing we forgot to add the Chinese version and reorganized the data cleaning step. It should also be removed later.

## 1_Data Cleaning:
This directory contains the original datasets used (MMLU datasets from OpenAI), as well as cleaning.ipynb, in which the cleaning process was conducted. The final datasets which will continue to be used are in 'mmlu_EN-US_balanced.csv' and 'mmlu_CH-ZH_balanced.csv' 

## 2_Order Bias:
This directory contains our first tests into observing order bias in different models. The cleaned data from the previous step is located in the 'data' directory, and the output of the tests is in the 'output' directory.

(right now there are very many order_bias files). The general concept is to ask the requested model each of the 1700 questions 4 times, with the correct answer being in each of the 4 possible positions (A, B, C, D). By comparing the performance with the 4 positions we can see whether the model is more likely to select a specific position as correct.

## 3_Approach 2:
This directory contains the code for our Cognitive Alignment approach to reducing order bias in MCQ questioning. Again, the data is contained in 'data' and the output is contained in 'output.'

The 'approach_2.py' file contains the specifics of enacting this approach. First, we use the LLM to answer the question as if it was a Free Response question (without considering any of the answer choices). Next, using the sentence-transformers/all-MiniLM-L6-v2 embedding model the LLM answer is embedded, as well as each of the multiple choice answers. Next, using cosine similarity the closest answer to the LLM answer is selected as the "correct" answer. The structure of this choice is completely detached from providing the LLM with the answer choices in a certain order, and therefore definitionally free from order bias :)