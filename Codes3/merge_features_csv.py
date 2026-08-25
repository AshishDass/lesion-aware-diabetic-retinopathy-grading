import pandas as pd

# Swin feature CSV (should contain 768 features + image_name)
swin_df = pd.read_csv("/Users/adass/Research/Codes3/swin_embeddings.csv")

# Lesion feature CSV (28 features + image_name)
lesion_df = pd.read_csv("/Users/adass/Research/Codes3/lesion_features_val.csv")
lesion_df = lesion_df.drop(columns=["flag_mild_DR", "flag_severe_NPDR", "flag_vision_threat"])
# Rename column
lesion_df = lesion_df.rename(columns={"image_name": "id_code"})
# Remove '.png' extension from id_code
lesion_df["id_code"] = lesion_df["id_code"].str.replace(".png", "", regex=False)
# lesion_df.to_csv("lesion_features_val.csv", index=False)
print("Dropped specified columns and saved the file.")

# Label CSV (image_name + label 0-4)
label_df = pd.read_csv("/Users/adass/Research/Codes3/labels.csv")

# # Merge all
df = swin_df.merge(lesion_df, on="id_code")
df = df.merge(label_df, on="id_code")
df.to_csv("/Users/adass/Research/Codes3/merged_features_labels.csv", index=False)
