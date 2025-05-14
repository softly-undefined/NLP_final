# this file cleans our test.csv data which is taken from https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv
# and maps it to the subcategories from https://github.com/hendrycks/test/blob/master/categories.py
# Creating a new dataset with 100 examples from each subcategory


import pandas as pd
import random

# Load your dataset
df = pd.read_csv("test.csv")

# Define subcategory mapping
subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

# Map each subject to a subcategory
df["Subcategory"] = df["Subject"].map(lambda x: subcategories.get(x, ["unknown"])[0])

# Sample 100 rows per subcategory (only if enough data exists)
TARGET = 100
grouped = df.groupby("Subcategory")
balanced_df = (
    grouped.filter(lambda x: len(x) >= TARGET)
           .groupby("Subcategory", group_keys=False)
           .apply(lambda x: x.sample(n=TARGET, random_state=42))
           .reset_index(drop=True)
)

# Save to CSV
balanced_df.to_csv("data.csv", index=False)

# Print summary
print("Subcategories included and number of examples:")
print(balanced_df['Subcategory'].value_counts())
print(f"\nTotal examples in data.csv: {len(balanced_df)}")
