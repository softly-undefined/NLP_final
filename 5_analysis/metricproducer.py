import pandas as pd
import numpy as np

# After loading your DataFrame into df and defining file_name:

file_list = ['order_bias_gpt-4o_zh', 'order_bias_gpt-4o_en', 'order_bias_mistral-small_en', 'order_bias_mistral-small_zh', 'order_bias_mistral-small_cot_en', 'order_bias_mistral-small_cot_zh', 'new_0.2_order_bias_mistral-small_cot_en', 'new_0.2_order_bias_mistral-small_cot_zh', 'new_0.459_order_bias_mistral-small_cot_en', 'new_0.459_order_bias_mistral-small_cot_zh']

for model_name in file_list:

    df = pd.read_csv(f"data/{model_name}/{model_name}.csv")

    # 1. Mark each prediction as correct (1) or incorrect (0)
    df["IsCorrect"] = (df["Predicted"] == df["RotationPosition"]).astype(int)

    # 2. Compute overall accuracy
    overall_acc = df["IsCorrect"].mean()

    # 3. Compute accuracy by each rotation position (A, B, C, D)
    acc_by_pos = (
        df
        .groupby("RotationPosition")["IsCorrect"]
        .mean()
        .reindex(["A", "B", "C", "D"])
    )

    # 4. Compute Relative Standard Deviation of Accuracies (RSD)
    acc_values = acc_by_pos.values
    std_acc = np.std(acc_values, ddof=0)  # population standard deviation
    rsd = std_acc / overall_acc

    # 5. Compute recall by position (same as accuracy-by-position in this setup)
    recall_by_pos = acc_by_pos.copy()

    # 6. Compute Standard Deviation of Recalls (RStd)
    recall_values = recall_by_pos.values
    std_recalls = np.std(recall_values, ddof=0)  # population standard deviation
    RStd = std_recalls

    # 7. Prepare baseline predictions for Fluctuation Rate (using rotation A as reference)
    baseline = (
        df[df["RotationPosition"] == "A"]
        .set_index("OriginalIndex")["Predicted"]
        .rename("BaselinePred")
    )

    # 8. Join baseline back to the original DataFrame
    df2 = df.join(baseline, on="OriginalIndex")

    # 9. Mark whether each prediction differs from the baseline
    df2["Fluctuated"] = (df2["Predicted"] != df2["BaselinePred"]).astype(int)

    # 10. Compute Fluctuation Rate
    fluctuation_rate = df2["Fluctuated"].mean()

    # 11. Gather all metrics into a dictionary
    metrics = {
        "overall_accuracy": overall_acc,
        "accuracy_A": acc_by_pos["A"],
        "accuracy_B": acc_by_pos["B"],
        "accuracy_C": acc_by_pos["C"],
        "accuracy_D": acc_by_pos["D"],
        "RSD": rsd,
        "recall_A": recall_by_pos["A"],
        "recall_B": recall_by_pos["B"],
        "recall_C": recall_by_pos["C"],
        "recall_D": recall_by_pos["D"],
        "RStd": RStd,
        "FluctuationRate": fluctuation_rate
    }

    # 12. Convert to a single-row DataFrame
    metrics_df = pd.DataFrame([metrics])

    # 13. Save to CSV named "metric_<file_name>.csv"
    output_filename = f"output/metric_{model_name}.csv"
    metrics_df.to_csv(output_filename, index=False)