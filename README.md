# NLP_Final Project (we should think of a name for the paper!)
Cognitive Alignment: Reducing Order Bias in MCQ Answering

## 1_Data Cleaning:
This directory contains the original datasets used (MMLU datasets from OpenAI), as well as cleaning.ipynb, in which the cleaning process was conducted. The final datasets which will continue to be used are in 'mmlu_EN-US_balanced.csv' and 'mmlu_CH-ZH_balanced.csv' 

## 2_Order Bias:
This directory contains our first tests into observing order bias in different models. The cleaned data from the previous step is located in the 'data' directory, and the output of the tests is in the 'output' directory.

The general concept is to ask the requested model each of the 1700 questions 4 times, with the correct answer being in each of the 4 possible positions (A, B, C, D). By comparing the performance with the 4 positions we can see whether the model is more likely to select a specific position as correct.

## 3_Pure_CA_Approach:
This directory contains the code for our Pure Cognitive Alignment approach to reducing order bias in MCQ questioning. Again, the data is contained in 'data' and the output is contained in 'output.'

The 'ca_appraoch.py' file contains the specifics of enacting this approach. First, we use the LLM to answer the question as if it was a Free Response question (without considering any of the answer choices). Next, using the sentence-transformers/all-MiniLM-L6-v2 embedding model the LLM answer is embedded, as well as each of the multiple choice answers. Next, using cosine similarity the closest answer to the LLM answer is selected as the "correct" answer. The structure of this choice is completely detached from providing the LLM with the answer choices in a certain order, and therefore definitionally free from order bias :)

## 4_CA-Adjusted_Approach
This directory contains the code for the CA-Adjusted MCQ Answering method. The code to replace certain answer choices with "DO NOT PICK THIS OPTION" is located in 'ca_adjusted_approach.py,'

The resultant CA-Adjusted datasets are then used in '2_order_bias' to generate order bias results influenced by the removed answer choices.


## 5_Analysis
This directory contains the code to calculate the RSD, RStd, and Fluctuation Rate metrics identified in the paper. The generation of metrics is located in 'metricproducer.py', the outputs of multiple files are created in 'outputcompilation.py,' and as usual the data is located in 'data' and ouput in 'output.'