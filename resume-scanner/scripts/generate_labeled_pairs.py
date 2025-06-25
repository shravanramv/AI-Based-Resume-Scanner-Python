import os
import glob
import random
import pandas as pd

# Path to resume root
RESUME_DIR = os.path.join("data", "resumes")

# Collect all resumes by category
category_to_files = {}
for category in os.listdir(RESUME_DIR):
    cat_path = os.path.join(RESUME_DIR, category)
    if os.path.isdir(cat_path):
        files = glob.glob(os.path.join(cat_path, "*.pdf"))
        if files:
            category_to_files[category] = files

# Generate pairs
positive_pairs = []
negative_pairs = []

for category, files in category_to_files.items():
    # POSITIVE: pair resumes from same category
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            positive_pairs.append((files[i], files[j], 1))  # label 1

    # NEGATIVE: pair with random files from different categories
    other_categories = [c for c in category_to_files if c != category]
    for file in files:
        neg_cat = random.choice(other_categories)
        neg_file = random.choice(category_to_files[neg_cat])
        negative_pairs.append((file, neg_file, 0))  # label 0

# Combine & shuffle
all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)

# Save to CSV
df = pd.DataFrame(all_pairs, columns=["resume_path", "jd_path", "label"])
df.to_csv("labeled_pairs.csv", index=False)
print(f"✅ Generated {len(df)} labeled pairs. Saved to labeled_pairs.csv.")
