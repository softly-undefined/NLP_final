import pandas as pd
import numpy as np

file_list = ['order_bias_gpt-4o_zh', 'order_bias_gpt-4o_en', 'order_bias_mistral-small_en', 'order_bias_mistral-small_zh', 'order_bias_mistral-small_cot_en', 'order_bias_mistral-small_cot_zh', 'new_0.2_order_bias_mistral-small_cot_en', 'new_0.2_order_bias_mistral-small_cot_zh', 'new_0.459_order_bias_mistral-small_cot_en', 'new_0.459_order_bias_mistral-small_cot_zh']

full_df = pd.DataFrame(columns=["model_name", "overall_accuracy", "accuracy_A", "accuracy_B", "accuracy_C", "accuracy_D", 
                                "RSD", "recall_A", "recall_B", "recall_C", "recall_D", "RStd", "FluctuationRate"])

for file in file_list:
    df = pd.read_csv(f"output/metric_{file}.csv")
    
    # Assuming the columns are named as specified
    full_df.loc[len(full_df)] = [file, 
                                 df['overall_accuracy'].iloc[0],  # overall_accuracy
                                 df['accuracy_A'].iloc[0],  # accuracy_A
                                 df['accuracy_B'].iloc[0],  # accuracy_B
                                 df['accuracy_C'].iloc[0],  # accuracy_C
                                 df['accuracy_D'].iloc[0],  # accuracy_D
                                 df['RSD'].iloc[0],  # RSD
                                 df['recall_A'].iloc[0],  # recall_A
                                 df['recall_B'].iloc[0],  # recall_B
                                 df['recall_C'].iloc[0],  # recall_C
                                 df['recall_D'].iloc[0],  # recall_D
                                 df['RStd'].iloc[0],  # RStd
                                 df['FluctuationRate'].iloc[0]]  # FluctuationRate

full_df.to_csv("output/fullmetrics.csv", index=False)
