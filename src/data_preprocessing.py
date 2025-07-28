# src/data_preprocessing.py
import re

def clean_text(text):
    text = str(text)
    text = re.sub(r'[^\w\s፡።]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_amharic(text):
    return text.split()
