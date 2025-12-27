from datasets import load_dataset
import pandas as pd
import random

dataset_name = "InferencePrince555/Resume-Dataset"
subset_name = None  
output_csv_path = "sampled_resumes.csv"
num_samples = 100 # number of resumes to randomly download

ds = load_dataset(dataset_name, split=subset_name or "train")

total_len = len(ds)
print(f"Total resumes available: {total_len}")

num_samples = min(num_samples, total_len)
sampled_indices = random.sample(range(total_len), num_samples)
sampled_data = ds.select(sampled_indices)


df = pd.DataFrame(sampled_data.to_dict())
df.to_csv(output_csv_path, index=False)
print(f"Saved {num_samples} random resumes to: {output_csv_path}")
