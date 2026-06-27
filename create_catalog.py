import pandas as pd

DISPOSITION_MAP = {
    "PC": 0,
    "FP": 1,
    "CP": 2,
    "KP": 3,
    "APC": 4,
    "FA": 5,
}

CLASS_NAMES = {v: k for k, v in DISPOSITION_MAP.items()}

df = pd.read_csv("data/raw/planet_training.csv")
print(f"Loaded {len(df)} rows from planet_training.csv")

df["Class_Label"] = df["TFOPWG Disposition"].map(DISPOSITION_MAP)

missing = df["Class_Label"].isna().sum()
if missing > 0:
    unknown = df.loc[df["Class_Label"].isna(), "TFOPWG Disposition"].unique()
    print(f"WARNING: {missing} rows with unknown disposition: {unknown}")
    df = df.dropna(subset=["Class_Label"])

df["Class_Label"] = df["Class_Label"].astype(int)

catalog = df[["TIC ID", "Period (days)", "Transit Epoch (BJD)", "Class_Label"]].copy()
catalog.columns = ["TIC_ID", "Period", "Epoch", "Class_Label"]

print(f"\nClass distribution:")
print(catalog["Class_Label"].value_counts().sort_index())
for cls, count in catalog["Class_Label"].value_counts().sort_index().items():
    print(f"  {cls} ({CLASS_NAMES[cls]}): {count}")

catalog.to_csv("data/processed/master_training_catalog_6class.csv", index=False)
print(f"\nSaved {len(catalog)} rows to data/processed/master_training_catalog_6class.csv")
