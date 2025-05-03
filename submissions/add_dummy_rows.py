import pandas as pd

TIMESTAMP = "0305051105"

input_csv_path = f"submissions/submission_{TIMESTAMP}.csv"
output_csv_path = f"submissions/submission_{TIMESTAMP}.csv"
test_csv_path = "nexar-collision-prediction/test.csv"

test_df = pd.read_csv(test_csv_path)
test_df["id"] = test_df["id"].apply(lambda x: str(x).zfill(5))
test_ids = test_df["id"].tolist()
target_rows = len(test_ids)

submission_df = pd.read_csv(input_csv_path)
submission_df["id"] = submission_df["id"].astype(str).str.zfill(5)

existing_ids = set(submission_df["id"])
missing_ids = [vid for vid in test_ids if vid not in existing_ids]

if missing_ids:
    padding = pd.DataFrame({
        'id': missing_ids,
        'score': [0.0] * len(missing_ids)
    })
    submission_df = pd.concat([submission_df, padding], ignore_index=True)
    print(f"Padded with {len(missing_ids)} missing video IDs.")
else:
    print(f"No padding needed. All {target_rows} IDs present.")

submission_df = submission_df.sort_values("id").reset_index(drop=True)
submission_df.to_csv(output_csv_path, index=False)
print(f"Saved padded CSV to {output_csv_path}")
