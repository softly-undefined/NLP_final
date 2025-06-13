import pandas as pd
import matplotlib.pyplot as plt

df2 = pd.read_csv('data/alignment_eval_mistral-small_cn.csv')
df = pd.read_csv('data/alignment_eval_mistral-small_en.csv')

import pandas as pd
import ast

CUTOFF = 0.459

def mark_wrong_options(row):
    scores = ast.literal_eval(row["SimilarityScores"])
    
    cutoff = max(scores) - CUTOFF
    
    opts = ["A", "B", "C", "D"]
    values = [row[opt] for opt in opts]
    
    new_values = [
        val if score >= cutoff else "DO NOT PICK THIS OPTION"
        for val, score in zip(values, scores)
    ]
    
    for opt, new_val in zip(opts, new_values):
        row[opt] = new_val
    
    return row

df = df.apply(mark_wrong_options, axis=1)
df2 = df2.apply(mark_wrong_options, axis=1)

df.to_csv(f"output/new_{CUTOFF}_alignment_eval_mistral-small_en.csv", index=False)
df2.to_csv(f"output/new_{CUTOFF}_alignment_eval_mistral-small_cn.csv", index=False)