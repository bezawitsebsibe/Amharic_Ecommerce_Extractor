
import pandas as pd
import random

# Load messages
df = pd.read_csv("data/telegram_data.csv")

# Drop duplicates and NaNs
messages = df["text"].dropna().drop_duplicates().tolist()

# Sample 40 messages
sampled_messages = random.sample(messages, 40)

# Save to a text file for manual labeling
with open("data/sample_messages_for_labeling.txt", "w", encoding="utf-8") as f:
    for i, msg in enumerate(sampled_messages):
        f.write(f"# Message {i+1}:\n{msg.strip()}\n\n")
