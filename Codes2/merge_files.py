import pandas as pd

# # Load the two CSV files
# file1 = pd.read_csv('/Users/adass/Research/archive/train_1.csv')
# file2 = pd.read_csv('/Users/adass/Research/archive/valid.csv')

# # Merge the files vertically
# merged_data = pd.concat([file1, file2], ignore_index=True)

# # Save the merged data to a new CSV file
# merged_data.to_csv('/Users/adass/Research/archive/train.csv', index=False)
df = pd.read_csv('/Users/adass/Research/archive/train.csv')
print(df.info())