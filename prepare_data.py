import pandas as pd
from sklearn.model_selection import train_test_split

# Read the CSV file
df = pd.read_csv('mortality_data.csv')

# Select and rename columns
df = df[['feature', 'labels']]
df.rename(columns={'feature': 'sequence','labels':'label'}, inplace=True)

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# Split the dataset
train, temp = train_test_split(df, test_size=0.1, random_state=42)
dev, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save the datasets into CSV files
train.to_csv('healthdata/train.csv', index=False)
dev.to_csv('healthdata/dev.csv', index=False)
test.to_csv('healthdata/test.csv', index=False)

print("Data split and saved successfully.")

