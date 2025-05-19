from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

app = FastAPI()
MODEL_PATH = "outputs/finetuned-mistral-lora"

class Headline(BaseModel):
    text: str

# Load once
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

@app.post("/score")
def score_headline(item: Headline):
    inputs = tokenizer(item.text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1).tolist()[0]
    labels = ["negative", "neutral", "positive"]
    return {labels[i]: probs[i] for i in range(len(probs))}
