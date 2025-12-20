import kagglehub
import shap
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


MODEL_NAME = "XSY/albert-base-v2-fakenews-discriminator"
DATASET_NAME = "clmentbisaillon/fake-and-real-news-dataset"


dataset_path = kagglehub.dataset_download(DATASET_NAME)


fake_dataset = pd.read_csv(dataset_path+"/Fake.csv")
real_dataset = pd.read_csv(dataset_path+"/True.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

torch.set_printoptions(precision=3, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)

def get_model_prediction_from_tokens(tokens):
    with torch.no_grad():
        output = model(**tokens)
        probs = torch.softmax(output.logits, dim=-1)
        return probs.numpy()


def get_model_prediction(texts):
    tokens = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
    return get_model_prediction_from_tokens(tokens)


batch_of_texts = list(fake_dataset["title"][:3]) + list(real_dataset["title"][:3])

preds = get_model_prediction(batch_of_texts)
print(preds)


shap_explainer = shap.Explainer(get_model_prediction, tokenizer)
shap_values = shap_explainer(batch_of_texts)
plots_html = shap.plots.text(shap_values, display=False)

with open("shap_text.html", "w", encoding="utf-8") as f:
    f.write(plots_html)

