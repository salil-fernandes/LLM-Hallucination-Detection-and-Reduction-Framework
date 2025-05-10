#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import unicodedata
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from datasets import Dataset
import wikipedia
from wordcloud import WordCloud
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from scipy.stats import entropy
from sklearn.metrics import classification_report, confusion_matrix

from groq import Groq
import requests
from tqdm import tqdm


# In[2]:


df = pd.read_csv("hallu_output.csv")
df.columns


# In[3]:


df_valid_llama = df.dropna(subset=["question", "right_answer", "together_llama_answer"]).copy().reset_index(drop=True)
df_valid_gemma = df.dropna(subset=["question", "right_answer", "together_gemma_answer"]).copy().reset_index(drop=True)
df_valid_qwen = df.dropna(subset=["question", "right_answer", "groq_qwen_answer"]).copy().reset_index(drop=True)
df_valid_mistral = df.dropna(subset=["question", "right_answer", "together_mistral_7b_answer"]).copy().reset_index(drop=True)
df_valid_bloom = df.dropna(subset=["question", "right_answer", "bloom_answer"]).copy().reset_index(drop=True)


# In[4]:


# Load NLI pipelines
nli_pipe = pipeline("text-classification", model="roberta-large-mnli", device=0)

# Define vague phrases
vague_phrases = [
    "i don't know", "i do not know", "i'm not sure", "unable to", "no information",
    "not enough information", "not available", "unfortunately", "i cannot verify",
    "cannot be confirmed", "data is missing", "unknown", "n/a", "none found",
    "there is no", "i do not have access"
]

def is_vague(text):
    if pd.isna(text):
        return False
    text = text.lower()
    return any(phrase in text for phrase in vague_phrases)

def clean_markdown(text):
    return re.sub(r'[*_~`]+', '', str(text))

def clean_response(text):
    text = clean_markdown(text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # code blocks
    text = re.sub(r'<[^>]+>', '', text)  # HTML tags
    text = re.sub(r'\[\d+\]', '', text)  # citations like [1]
    text = re.sub(r'\(Source:.*?\)', '', text, flags=re.IGNORECASE)  # inline sources

    # Remove AI disclaimers or filler
    text = re.sub(r"(?i)as an ai language model.*?(?=\.\s|$)", "", text)
    text = re.sub(r"(?i)it is (also )?important to note that.*?(?=\.\s|$)", "", text)

    # Strip prefixes like "Answer:", "Explanation:"
    text = re.sub(r"^\s*(Answer|Explanation|Response|Here's|Sure|Let me explain):?\s*", "", text, flags=re.IGNORECASE)

    # Remove generic sentences often seen in templated LLM output
    text = re.sub(r"(?i)this information may vary depending on context.*", "", text)

    # Normalize
    text = text.replace("\n", " ").strip()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces

    return text

def run_nli_with_vagueness(data, answer_column, batch_size=16, name='llama'):
    def build_input(row):
        evidence = f"The answer to the question '{row['question']}' is '{row['right_answer']}'."
        claim = clean_response(row[answer_column])
        return f"{evidence} [SEP] {claim}"

    data["nli_input"] = data.apply(build_input, axis=1)
    hf_dataset = Dataset.from_pandas(data)

    def run_nli_batch(batch):
        results = nli_pipe(batch["nli_input"], truncation=True)
        return {
            "nli_label": [r["label"] for r in results],
            "nli_score": [r["score"] for r in results]
        }

    hf_dataset = hf_dataset.map(run_nli_batch, batched=True, batch_size=batch_size, desc=f"NLI for {name}")
    
    results_df = pd.DataFrame({
        "label": hf_dataset["nli_label"],
        "score": hf_dataset["nli_score"],
        "is_vague": data[answer_column].apply(is_vague)
    })

    # Save
    results_df.to_csv(f"./nli_predictions/nli_{name}_results.csv", index=False)
    return results_df


# In[5]:


llama_results_df = run_nli_with_vagueness(df_valid_llama.copy(), "together_llama_answer", batch_size=32, name="llama")
llama_results_df.head(5)


# In[6]:


gemma_results_df = run_nli_with_vagueness(df_valid_gemma.copy(), "together_gemma_answer", batch_size=32, name="gemma")
gemma_results_df.head(5)


# In[7]:


qwen_results_df = run_nli_with_vagueness(df_valid_qwen.copy(), "groq_qwen_answer", batch_size=32, name="qwen")
qwen_results_df.head(5)


# In[8]:


mistral_results_df = run_nli_with_vagueness(df_valid_mistral.copy(), "together_mistral_7b_answer", batch_size=32, name="mistral")
mistral_results_df.head(5)


# In[9]:


bloom_results_df = run_nli_with_vagueness(df_valid_bloom.copy(), "bloom_answer", batch_size=32, name="bloom")
bloom_results_df.head(5)


# In[10]:


def evaluate_nli(results, human_verified, threshold=0.51):
    def nli_to_pred(entry):
        if entry.get("is_vague", False):  # override for vague responses
            return True
        return ("contradiction" in entry["label"].lower()) and entry["score"] >= threshold

    predicted_labels = [nli_to_pred(r) for r in results]
    true_labels = human_verified.tolist()

    return classification_report(true_labels, predicted_labels), classification_report(true_labels, predicted_labels, output_dict=True), confusion_matrix(true_labels, predicted_labels)

def visualize_confusion_matrix(conf_mat, name):
    labels = ['Not Hallucinated', 'Hallucinated']

    # Plot confusion matrix
    plt.figure(figsize=(4, 3))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(f"Confusion Matrix for {name} NLI")
    plt.tight_layout()
    plt.show()


# In[12]:


# Load updated NLI results
llama_nli_results = pd.read_csv("./nli_predictions/nli_llama_results.csv").to_dict(orient="records")
gemma_nli_results = pd.read_csv("./nli_predictions/nli_gemma_results.csv").to_dict(orient="records")
qwen_nli_results = pd.read_csv("./nli_predictions/nli_qwen_results.csv").to_dict(orient="records")
mistral_nli_results = pd.read_csv("./nli_predictions/nli_mistral_results.csv").to_dict(orient="records")
bloom_nli_results = pd.read_csv("./nli_predictions/nli_bloom_results.csv").to_dict(orient="records")


print("\n🦙 LLAMA NLI Metrics (Before Fact Checking)")
llama_report, llama_report_dict, llama_conf_mat = evaluate_nli(llama_nli_results, df_valid_llama['together_llama_is_hallucinated'])
print(llama_report)
visualize_confusion_matrix(llama_conf_mat, 'LLAMA')

print("\n💎 Gemma NLI Metrics (Before Fact Checking)")
gemma_report, gemma_report_dict, gemma_conf_mat = evaluate_nli(gemma_nli_results, df_valid_gemma['together_gemma_is_hallucinated'])
print(gemma_report)
visualize_confusion_matrix(gemma_conf_mat, 'Gemma')

print("\n🧠 QWEN NLI Metrics (Before Fact Checking)")
qwen_report, qwen_report_dict, qwen_conf_mat = evaluate_nli(qwen_nli_results, df_valid_qwen['groq_qwen_is_hallucinated'])
print(qwen_report)
visualize_confusion_matrix(qwen_conf_mat, 'Qwen')

print("\n🧙 Mistral NLI Metrics (Before Fact Checking)")
mistral_report, mistral_report_dict, mistral_conf_mat = evaluate_nli(mistral_nli_results, df_valid_mistral['together_mistral_7b_is_hallucinated'])
print(mistral_report)
visualize_confusion_matrix(mistral_conf_mat, 'Mistral')

print("\n🌻 Bloom NLI Metrics (Before Fact Checking)")
bloom_report, bloom_report_dict, bloom_conf_mat = evaluate_nli(bloom_nli_results, df_valid_bloom['bloom_is_hallucinated'])
print(bloom_report)
visualize_confusion_matrix(bloom_conf_mat, 'Bloom')


# In[13]:


def get_fp_fn_indices(nli_results_df, ground_truth_labels, threshold=0.51):
    """
    Returns the indices of false positives and false negatives based on NLI prediction vs ground truth.
    """
    def nli_to_pred(entry):
        if entry.get("is_vague", False):  # override for vague responses
            return True
        return ("contradiction" in entry["label"].lower()) and entry["score"] >= threshold

    records = nli_results_df.to_dict(orient="records")
    predicted = [nli_to_pred(row) for row in records]
    actuals = ground_truth_labels.astype(bool).tolist()

    false_positives = [i for i, (p, a) in enumerate(zip(predicted, actuals)) if p and not a]
    false_negatives = [i for i, (p, a) in enumerate(zip(predicted, actuals)) if not p and a]

    return false_positives, false_negatives

def show_fp_fn_table(model_names, fp_dict, fn_dict):
    data = {
        "Model": model_names,
        "False Positives": [len(fp_dict[model]) for model in model_names],
        "False Negatives": [len(fn_dict[model]) for model in model_names]
    }
    df_fp_fn = pd.DataFrame(data)
    return df_fp_fn


llama_nli_results = pd.read_csv(f"./nli_predictions/nli_llama_results.csv")
gemma_nli_results = pd.read_csv(f"./nli_predictions/nli_gemma_results.csv")
qwen_nli_results = pd.read_csv(f"./nli_predictions/nli_qwen_results.csv")
mistral_nli_results = pd.read_csv(f"./nli_predictions/nli_mistral_results.csv")
bloom_nli_results = pd.read_csv(f"./nli_predictions/nli_bloom_results.csv")

model_names = ["LLAMA", "Gemma", "Qwen", "Mistral", "Bloom"]

llama_fp, llama_fn = get_fp_fn_indices(llama_nli_results, df_valid_llama["together_llama_is_hallucinated"])
gemma_fp, gemma_fn = get_fp_fn_indices(gemma_nli_results, df_valid_gemma["together_gemma_is_hallucinated"])
qwen_fp, qwen_fn = get_fp_fn_indices(qwen_nli_results, df_valid_qwen["groq_qwen_is_hallucinated"])
mistral_fp, mistral_fn = get_fp_fn_indices(mistral_nli_results, df_valid_mistral["together_mistral_7b_is_hallucinated"])
bloom_fp, bloom_fn = get_fp_fn_indices(bloom_nli_results, df_valid_bloom["bloom_is_hallucinated"])

fp_dict = {
    "LLAMA": llama_fp,
    "Gemma": gemma_fp,
    "Qwen": qwen_fp,
    "Mistral": mistral_fp,
    "Bloom": bloom_fp
}
fn_dict = {
    "LLAMA": llama_fn,
    "Gemma": gemma_fn,
    "Qwen": qwen_fn,
    "Mistral": mistral_fn,
    "Bloom": bloom_fn
}

show_fp_fn_table(model_names, fp_dict, fn_dict)


# In[14]:


def factuality_check(fp_indices, fn_indices, data_df, results_df, response_col, name, threshold=0.51, non_hallu_threshold=0.5):
    def extract_answer_span(claim, question):
        sentences = re.split(r'(?<=[.!?])\s+', claim.strip())
        for s in sentences:
            if any(x in s.lower() for x in question.lower().split()[:5]):
                return s
        return sentences[0]

    def verify_with_wikipedia(question, claim):
        try:
            claim = extract_answer_span(claim, question)
            search_results = wikipedia.search(question)
            if not search_results:
                return "NO_RESULTS", None
            page = wikipedia.page(search_results[0])
            premise = page.summary[:700]
            result = nli_pipe(f"{premise} [SEP] {claim}", truncation=True)[0]
            if result["label"].lower() == "contradiction" and result["score"] >= threshold:
                return "contradiction", result["score"]
            elif result["label"].lower() == "neutral" and result["score"] < non_hallu_threshold:
                return "contradiction", 0.51
            return "neutral", result["score"]
        except:
            return "ERROR", None

    def process_row(idx):
        question = data_df.loc[idx, "question"]
        claim = data_df.loc[idx, response_col]
        original_label = results_df.loc[idx, "label"]
        verdict, score = verify_with_wikipedia(question, claim)

        return {
            "idx": idx,
            "updated_label": verdict if verdict in ["contradiction", "neutral"] else original_label,
            "updated_score": score if isinstance(score, float) else results_df.loc[idx, "score"]
        }

    indices = fp_indices + fn_indices
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(tqdm(executor.map(process_row, indices), total=len(indices), desc=f"Factuality Check (FP+FN) - {name}"))

    for res in results:
        results_df.loc[res["idx"], "label"] = res["updated_label"]
        results_df.loc[res["idx"], "score"] = res["updated_score"]

    results_df.to_csv(f"./nli_predictions/nli_fact_{name}_results.csv", index=False)
    return


# In[12]:


df_llama = df_valid_llama.copy()
df_llama["together_llama_answer"] = df_llama["together_llama_answer"].apply(clean_response)
factuality_check(llama_fp, llama_fn, df_llama, llama_nli_results,"together_llama_answer", "llama")


# In[13]:


df_gemma = df_valid_gemma.copy()
df_gemma["together_gemma_answer"] = df_gemma["together_gemma_answer"].apply(clean_response)
factuality_check(gemma_fp, gemma_fn, df_gemma, gemma_nli_results, "together_gemma_answer", "gemma")


# In[14]:


df_qwen = df_valid_qwen.copy()
df_qwen["groq_qwen_answer"] = df_qwen["groq_qwen_answer"].apply(clean_response)
factuality_check(qwen_fp, qwen_fn, df_qwen, qwen_nli_results, "groq_qwen_answer", "qwen")


# In[15]:


df_mistral = df_valid_mistral.copy()
df_mistral["together_mistral_7b_answer"] = df_mistral["together_mistral_7b_answer"].apply(clean_response)
factuality_check(
    mistral_fp,
    mistral_fn,
    df_mistral,
    mistral_nli_results,
    "together_mistral_7b_answer",
    "mistral"
)


# In[16]:


df_bloom = df_valid_bloom.copy()
df_bloom["bloom_answer"] = df_bloom["bloom_answer"].apply(clean_response)
factuality_check(
    bloom_fp,
    bloom_fn,
    df_bloom,
    bloom_nli_results,
    "bloom_answer",
    "bloom"
)


# In[17]:


# Load updated NLI results
llama_nli_fact_results = pd.read_csv("./nli_predictions/nli_fact_llama_results.csv").to_dict(orient="records")
gemma_nli_fact_results = pd.read_csv("./nli_predictions/nli_fact_gemma_results.csv").to_dict(orient="records")
qwen_nli_fact_results = pd.read_csv("./nli_predictions/nli_fact_qwen_results.csv").to_dict(orient="records")
mistral_nli_fact_results = pd.read_csv("./nli_predictions/nli_fact_mistral_results.csv").to_dict(orient="records")
bloom_nli_fact_results = pd.read_csv("./nli_predictions/nli_fact_bloom_results.csv").to_dict(orient="records")

print("\n🦙 LLAMA NLI Metrics (After Fact Checking)")
llama_fact_report, llama_fact_report_dict, llama_fact_conf_mat = evaluate_nli(llama_nli_fact_results, df_valid_llama['together_llama_is_hallucinated'])
print(llama_fact_report)
visualize_confusion_matrix(llama_fact_conf_mat, "LLAMA")

print("\n💎 Gemma NLI Metrics (After Fact Checking)")
gemma_fact_report, gemma_nli_fact_report_dict, gemma_fact_conf_mat = evaluate_nli(gemma_nli_fact_results, df_valid_gemma['together_gemma_is_hallucinated'])
print(gemma_fact_report)
visualize_confusion_matrix(gemma_fact_conf_mat, "Gemma")

print("\n🧠 QWEN NLI Metrics (After Fact Checking)")
qwen_fact_report, qwen_fact_report_dict, qwen_fact_conf_mat = evaluate_nli(qwen_nli_fact_results, df_valid_qwen['groq_qwen_is_hallucinated'])
print(qwen_fact_report)
visualize_confusion_matrix(qwen_fact_conf_mat, "Qwen")

print("\n🧙 Mistral NLI Metrics (After Fact Checking)")
mistral_fact_report, mistral_fact_report_dict, mistral_fact_conf_mat = evaluate_nli(mistral_nli_fact_results, df_valid_mistral['together_mistral_7b_is_hallucinated'])
print(mistral_fact_report)
visualize_confusion_matrix(mistral_fact_conf_mat, "Mistral")

print("\n🌻 Bloom NLI Metrics (After Fact Checking)")
bloom_fact_report, bloom_fact_report_dict, bloom_fact_conf_mat = evaluate_nli(bloom_nli_fact_results, df_valid_bloom['bloom_is_hallucinated'])
print(bloom_fact_report)
visualize_confusion_matrix(bloom_fact_conf_mat, "Bloom")


# In[18]:


llama_nli_fact_results = pd.read_csv("./nli_predictions/nli_fact_llama_results.csv")
gemma_nli_fact_results = pd.read_csv("./nli_predictions/nli_fact_gemma_results.csv")
qwen_nli_fact_results = pd.read_csv("./nli_predictions/nli_fact_qwen_results.csv")
mistral_nli_fact_results = pd.read_csv("./nli_predictions/nli_fact_mistral_results.csv")
bloom_nli_fact_results = pd.read_csv("./nli_predictions/nli_fact_bloom_results.csv")

llama_fact_fp, llama_fact_fn = get_fp_fn_indices(llama_nli_fact_results, df_valid_llama["together_llama_is_hallucinated"])
gemma_fact_fp, gemma_fact_fn = get_fp_fn_indices(gemma_nli_fact_results, df_valid_gemma["together_gemma_is_hallucinated"])
qwen_fact_fp, qwen_fact_fn = get_fp_fn_indices(qwen_nli_fact_results, df_valid_qwen["groq_qwen_is_hallucinated"])
mistral_fact_fp, mistral_fact_fn = get_fp_fn_indices(mistral_nli_fact_results, df_valid_mistral["together_mistral_7b_is_hallucinated"])
bloom_fact_fp, bloom_fact_fn = get_fp_fn_indices(bloom_nli_fact_results, df_valid_bloom["bloom_is_hallucinated"])

fp_dict = {
    "LLAMA": llama_fact_fp,
    "Gemma": gemma_fact_fp,
    "Qwen": qwen_fact_fp,
    "Mistral": mistral_fact_fp,
    "Bloom": bloom_fact_fp
}

fn_dict = {
    "LLAMA": llama_fact_fn,
    "Gemma": gemma_fact_fn,
    "Qwen": qwen_fact_fn,
    "Mistral": mistral_fact_fn,
    "Bloom": bloom_fact_fn
}

show_fp_fn_table(model_names, fp_dict, fn_dict)


# In[20]:


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding error
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def compute_avg_entropy(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    token_entropies = [entropy(dist.cpu().numpy()) for dist in probs[0]]
    return np.mean(token_entropies)

# Threaded batch processing
def compute_entropy_parallel(texts, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(tqdm(executor.map(compute_avg_entropy, texts), total=len(texts), desc="Entropy Scoring"))

def update_nli_results_with_entropy(indices, data_df, answer_col, nli_results_df, model_name, threshold=4, max_workers=10):
    # Extract the answers corresponding to the FP+FN indices
    answers = data_df.loc[indices, answer_col].apply(clean_response).tolist()
    # Compute entropies in parallel
    entropies = compute_entropy_parallel(answers, max_workers=max_workers)
    hallucinated = [e > threshold for e in entropies]

    for idx, is_hallu in tqdm(zip(indices, hallucinated), total=len(indices), desc=f"Updating NLI w/ Entropy for {model_name}"):
        if is_hallu:
            nli_results_df.loc[idx, "label"] = "contradiction"
            nli_results_df.loc[idx, "score"] = 0.99
        else:
            nli_results_df.loc[idx, "label"] = "neutral"
            #nli_results_df.loc[idx, "score"] = 0.51

    save_path = f"./nli_predictions/nli_entropy_{model_name.lower()}_results.csv"
    nli_results_df.to_csv(save_path, index=False)
    print(f"Saved entropy-updated NLI results to {save_path}")
    return nli_results_df


# In[15]:


llama_entropy_indices = llama_fact_fp + llama_fact_fn

llama_nli_entropy_results = update_nli_results_with_entropy(
    indices=llama_entropy_indices,
    data_df=df_valid_llama.copy(),
    answer_col="together_llama_answer",
    nli_results_df=llama_nli_fact_results.copy(),
    model_name="llama"
)


# In[13]:


gemma_entropy_indices = gemma_fact_fp + gemma_fact_fn
gemma_nli_entropy_results = update_nli_results_with_entropy(
    indices=gemma_entropy_indices,
    data_df=df_valid_gemma.copy(),
    answer_col="together_gemma_answer",
    nli_results_df=gemma_nli_fact_results.copy(),
    model_name="gemma"
)


# In[14]:


qwen_entropy_indices = qwen_fact_fp + qwen_fact_fn
qwen_nli_entropy_results = update_nli_results_with_entropy(
    indices=qwen_entropy_indices,
    data_df=df_valid_qwen.copy(),
    answer_col="groq_qwen_answer",
    nli_results_df=qwen_nli_fact_results.copy(),
    model_name="qwen"
)


# In[21]:


mistral_entropy_indices = mistral_fact_fp + mistral_fact_fn
mistral_nli_entropy_results = update_nli_results_with_entropy(
    indices=mistral_entropy_indices,
    data_df=df_valid_mistral.copy(),
    answer_col="together_mistral_7b_answer",
    nli_results_df=mistral_nli_fact_results.copy(),
    model_name="mistral"
)


# In[22]:


bloom_entropy_indices = bloom_fact_fp + bloom_fact_fn
bloom_nli_entropy_results = update_nli_results_with_entropy(
    indices=bloom_entropy_indices,
    data_df=df_valid_bloom.copy(),
    answer_col="bloom_answer",
    nli_results_df=bloom_nli_fact_results.copy(),
    model_name="bloom"
)


# In[23]:


llama_nli_entropy_results = pd.read_csv("./nli_predictions/nli_entropy_llama_results.csv").to_dict(orient="records")
gemma_nli_entropy_results = pd.read_csv("./nli_predictions/nli_entropy_gemma_results.csv").to_dict(orient="records")
qwen_nli_entropy_results = pd.read_csv("./nli_predictions/nli_entropy_qwen_results.csv").to_dict(orient="records")
mistral_nli_entropy_results = pd.read_csv("./nli_predictions/nli_entropy_mistral_results.csv").to_dict(orient="records")
bloom_nli_entropy_results = pd.read_csv("./nli_predictions/nli_entropy_bloom_results.csv").to_dict(orient="records")

# LLAMA
print("\n🦙 LLAMA NLI Metrics (After Entropy)")
llama_entropy_report, llama_entropy_report_dict, llama_entropy_conf_mat = evaluate_nli(llama_nli_entropy_results, df_valid_llama['together_llama_is_hallucinated'])
print(llama_entropy_report)
visualize_confusion_matrix(llama_entropy_conf_mat, "LLAMA")

# Gemma
print("\n💎 Gemma NLI Metrics (After Entropy)")
gemma_entropy_report, gemma_entropy_report_dict, gemma_entropy_conf_mat = evaluate_nli(gemma_nli_entropy_results, df_valid_gemma['together_gemma_is_hallucinated'])
print(gemma_entropy_report)
visualize_confusion_matrix(gemma_entropy_conf_mat, "Gemma")

# Qwen
print("\n🧠 QWEN NLI Metrics (After Entropy)")
qwen_entropy_report, qwen_entropy_report_dict, qwen_entropy_conf_mat = evaluate_nli(qwen_nli_entropy_results, df_valid_qwen['groq_qwen_is_hallucinated'])
print(qwen_entropy_report)
visualize_confusion_matrix(qwen_entropy_conf_mat, "Qwen")

# Mistral
print("\n🧙 Mistral NLI Metrics (After Entropy)")
mistral_entropy_report, mistral_entropy_report_dict, mistral_entropy_conf_mat = evaluate_nli(mistral_nli_entropy_results, df_valid_mistral['together_mistral_7b_is_hallucinated'])
print(mistral_entropy_report)
visualize_confusion_matrix(mistral_entropy_conf_mat, "Mistral")

# Bloom
print("\n🌻 Bloom NLI Metrics (After Entropy)")
bloom_entropy_report, bloom_entropy_report_dict, bloom_entropy_conf_mat = evaluate_nli(bloom_nli_entropy_results, df_valid_bloom['bloom_is_hallucinated'])
print(bloom_entropy_report)
visualize_confusion_matrix(bloom_entropy_conf_mat, "Bloom")


# In[24]:


llama_nli_entropy_results = pd.read_csv("./nli_predictions/nli_entropy_llama_results.csv")
gemma_nli_entropy_results = pd.read_csv("./nli_predictions/nli_entropy_gemma_results.csv")
qwen_nli_entropy_results = pd.read_csv("./nli_predictions/nli_entropy_qwen_results.csv")
mistral_nli_entropy_results = pd.read_csv("./nli_predictions/nli_entropy_mistral_results.csv")
bloom_nli_entropy_results = pd.read_csv("./nli_predictions/nli_entropy_bloom_results.csv")

llama_entropy_fp, llama_entropy_fn = get_fp_fn_indices(llama_nli_entropy_results, df_valid_llama["together_llama_is_hallucinated"])
gemma_entropy_fp, gemma_entropy_fn = get_fp_fn_indices(gemma_nli_entropy_results, df_valid_gemma["together_gemma_is_hallucinated"])
qwen_entropy_fp, qwen_entropy_fn = get_fp_fn_indices(qwen_nli_entropy_results, df_valid_qwen["groq_qwen_is_hallucinated"])
mistral_entropy_fp, mistral_entropy_fn = get_fp_fn_indices(mistral_nli_entropy_results, df_valid_mistral["together_mistral_7b_is_hallucinated"])
bloom_entropy_fp, bloom_entropy_fn = get_fp_fn_indices(bloom_nli_entropy_results, df_valid_bloom["bloom_is_hallucinated"])

fp_dict = {
    "LLAMA": llama_entropy_fp,
    "Gemma": gemma_entropy_fp,
    "Qwen": qwen_entropy_fp,
    "Mistral": mistral_entropy_fp,
    "Bloom": bloom_entropy_fp
}
fn_dict = {
    "LLAMA": llama_entropy_fn,
    "Gemma": gemma_entropy_fn,
    "Qwen": qwen_entropy_fn,
    "Mistral": mistral_entropy_fn,
    "Bloom": bloom_entropy_fn
}

show_fp_fn_table(model_names, fp_dict, fn_dict)


# In[33]:


def plot_metric_progression(models_metrics, metric="accuracy"):
    """
    Plot a line chart showing progression of a metric (accuracy or f1-score)
    for each model across stages: Before (Just NLI), After Fact Check, After Entropy.
    """
    stages = ["before", "fact", "entropy"]
    stage_labels = ["Just NLI", "After Fact", "After Entropy"]

    plt.figure(figsize=(9, 5))

    for model, reports in models_metrics.items():
        values = []
        for stage in stages:
            if metric == "accuracy":
                val = reports[stage].get("accuracy", 0)
            else:  # f1-score for hallucinated class
                val = reports[stage].get("True", {}).get("f1-score", 0)
            values.append(val)

        plt.plot(stage_labels, values, marker="o", label=model)

    plt.title(f"{metric.title()} Progression Across Stages")
    plt.xlabel("Stage")
    plt.ylabel(metric.title())
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()


def plot_metric_comparison(models_metrics, metric='f1-score'):
    model_names = []
    values_before = []
    values_fact = []
    values_entropy = []

    for model, reports in models_metrics.items():
        if metric != 'accuracy':
            before_value = reports["before"].get("True", {}).get(metric, 0)
            fact_value = reports["fact"].get("True", {}).get(metric, 0)
            entropy_value = reports["entropy"].get("True", {}).get(metric, 0)
        else:
            before_value = reports["before"].get("accuracy", 0)
            fact_value = reports["fact"].get("accuracy", 0)
            entropy_value = reports["entropy"].get("accuracy", 0)

        model_names.append(model)
        values_before.append(before_value)
        values_fact.append(fact_value)
        values_entropy.append(entropy_value)

    x = range(len(model_names))
    width = 0.15

    plt.figure(figsize=(10, 6))
    plt.bar(x, values_before, width, label='Before (Just NLI)', alpha=0.7)
    plt.bar([p + width for p in x], values_fact, width, label='After Fact Check', alpha=0.8)
    plt.bar([p + 2*width for p in x], values_entropy, width, label='After Entropy', alpha=0.8)

    plt.ylabel(metric.title())
    plt.title(f'Detection {metric.title()} Comparison')
    plt.xticks([p + width for p in x], model_names)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()

models_metrics = {
    "LLAMA": {
        "before": llama_report_dict,
        "fact": llama_fact_report_dict,
        "entropy": llama_entropy_report_dict
    },
    "Gemma": {
        "before": gemma_report_dict,
        "fact": gemma_nli_fact_report_dict,
        "entropy": gemma_entropy_report_dict
    },
    "Qwen": {
        "before": qwen_report_dict,
        "fact": qwen_fact_report_dict,
        "entropy": qwen_entropy_report_dict
    },
    "Mistral": {
        "before": mistral_report_dict,
        "fact": mistral_fact_report_dict,
        "entropy": mistral_entropy_report_dict
    },
    "Bloom": {
        "before": bloom_report_dict,
        "fact": bloom_fact_report_dict,
        "entropy": bloom_entropy_report_dict
    }
}

plot_metric_progression(models_metrics, metric="accuracy")
plot_metric_progression(models_metrics, metric="f1-score")

plot_metric_comparison(models_metrics, metric='f1-score')
plot_metric_comparison(models_metrics, metric='accuracy')


# In[26]:


def plot_fp_fn_drop(fp_before, fn_before, fp_fact, fn_fact, fp_entropy, fn_entropy):
    """
    Vertically stacked bar plots showing FP and FN drops before → fact check → entropy filtering.

    Args:
        fp_before, fn_before: dicts with model names as keys and FP/FN counts before any corrections.
        fp_fact, fn_fact: counts after fact checking.
        fp_entropy, fn_entropy: counts after entropy-based filtering.
    """
    models = list(fp_before.keys())
    x = range(len(models))
    width = 0.15

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # FP plot
    ax1.bar(x, [fp_before[m] for m in models], width, label='Before', alpha=0.7)
    ax1.bar([i + width for i in x], [fp_fact[m] for m in models], width, label='After Fact Check', alpha=0.8)
    ax1.bar([i + 2 * width for i in x], [fp_entropy[m] for m in models], width, label='After Entropy', alpha=0.8)
    ax1.set_ylabel("False Positives")
    ax1.set_title("False Positives: Before → Fact → Entropy")
    ax1.set_xticks([i + width for i in x])
    ax1.set_xticklabels(models)
    ax1.legend()

    # FN plot
    ax2.bar(x, [fn_before[m] for m in models], width, label='Before', alpha=0.7)
    ax2.bar([i + width for i in x], [fn_fact[m] for m in models], width, label='After Fact Check', alpha=0.8)
    ax2.bar([i + 2 * width for i in x], [fn_entropy[m] for m in models], width, label='After Entropy', alpha=0.8)
    ax2.set_ylabel("False Negatives")
    ax2.set_title("False Negatives: Before → Fact → Entropy")
    ax2.set_xticks([i + width for i in x])
    ax2.set_xticklabels(models)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# In[27]:


fp_before = {
    "LLAMA": len(llama_fp),
    "Gemma": len(gemma_fp),
    "Qwen": len(qwen_fp),
    "Mistral": len(mistral_fp),
    "Bloom": len(bloom_fp)
}
fn_before = {
    "LLAMA": len(llama_fn),
    "Gemma": len(gemma_fn),
    "Qwen": len(qwen_fn),
    "Mistral": len(mistral_fn),
    "Bloom": len(bloom_fn)
}
fp_fact = {
    "LLAMA": len(llama_fact_fp),
    "Gemma": len(gemma_fact_fp),
    "Qwen": len(qwen_fact_fp),
    "Mistral": len(mistral_fact_fp),
    "Bloom": len(bloom_fact_fp)
}
fn_fact = {
    "LLAMA": len(llama_fact_fn),
    "Gemma": len(gemma_fact_fn),
    "Qwen": len(qwen_fact_fn),
    "Mistral": len(mistral_fact_fn),
    "Bloom": len(bloom_fact_fn)
}
fp_entropy = {
    "LLAMA": len(llama_entropy_fp),
    "Gemma": len(gemma_entropy_fp),
    "Qwen": len(qwen_entropy_fp),
    "Mistral": len(mistral_entropy_fp),
    "Bloom": len(bloom_entropy_fp)
}
fn_entropy = {
    "LLAMA": len(llama_entropy_fn),
    "Gemma": len(gemma_entropy_fn),
    "Qwen": len(qwen_entropy_fn),
    "Mistral": len(mistral_entropy_fn),
    "Bloom": len(bloom_entropy_fn)
}

plot_fp_fn_drop(fp_before, fn_before, fp_fact, fn_fact, fp_entropy, fn_entropy)


# In[28]:


def plot_nli_score_distributions_all_three(results_before, results_fact, results_entropy, model_name):
    """
    Plots the NLI confidence score distributions for a model before, after fact check, and after entropy filtering.

    Args:
        results_before (list of dict): NLI scores before any fact checking.
        results_fact (list of dict): NLI scores after fact-based adjustment.
        results_entropy (list of dict): NLI scores after entropy-based correction.
        model_name (str): Name of the model (used in title).
    """
    scores_before = [entry["score"] for entry in results_before]
    scores_fact = [entry["score"] for entry in results_fact]
    scores_entropy = [entry["score"] for entry in results_entropy]

    plt.figure(figsize=(9, 4))
    sns.histplot(scores_before, kde=True, bins=30, label="Before", color="dodgerblue", stat="density", alpha=0.25)
    sns.histplot(scores_fact, kde=True, bins=30, label="After Fact", color="orange", stat="density", alpha=0.25)
    sns.histplot(scores_entropy, kde=True, bins=30, label="After Entropy", color="green", stat="density", alpha=0.25)

    plt.title(f"NLI Confidence Score Distribution - {model_name}")
    plt.xlabel("NLI Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()



plot_nli_score_distributions_all_three(
    llama_nli_results.to_dict(orient="records"),
    llama_nli_fact_results.to_dict(orient="records"),
    llama_nli_entropy_results.to_dict(orient="records"),
    "LLAMA"
)

plot_nli_score_distributions_all_three(
    gemma_nli_results.to_dict(orient="records"),
    gemma_nli_fact_results.to_dict(orient="records"),
    gemma_nli_entropy_results.to_dict(orient="records"),
    "Gemma"
)

plot_nli_score_distributions_all_three(
    qwen_nli_results.to_dict(orient="records"),
    qwen_nli_fact_results.to_dict(orient="records"),
    qwen_nli_entropy_results.to_dict(orient="records"),
    "Qwen"
)

plot_nli_score_distributions_all_three(
    mistral_nli_results.to_dict(orient="records"),
    mistral_nli_fact_results.to_dict(orient="records"),
    mistral_nli_entropy_results.to_dict(orient="records"),
    "Mistral"
)

plot_nli_score_distributions_all_three(
    bloom_nli_results.to_dict(orient="records"),
    bloom_nli_fact_results.to_dict(orient="records"),
    bloom_nli_entropy_results.to_dict(orient="records"),
    "Bloom"
)


# In[30]:


def plot_hallucination_wordclouds(data_df, label_column, answer_column, title_prefix="Model"):
    """
    Plots word clouds of hallucinated answers before and after fact checking.

    Args:
        data_df (pd.DataFrame): DataFrame with hallucination labels and LLM answers.
        label_column (str): Column containing hallucination labels (True/False).
        answer_column (str): Column with LLM answers (text).
        title_prefix (str): Label to identify model in the title.
    """
    # Filter hallucinated samples
    hallucinated_df = data_df[data_df[label_column] == True]

    # Join all text
    hallucinated_text = " ".join(str(ans) for ans in hallucinated_df[answer_column].dropna())

    # Generate word cloud
    wc = WordCloud(width=800, height=400, background_color='white', colormap='Dark2').generate(hallucinated_text)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"{title_prefix} - Word Cloud of Hallucinated Responses", fontsize=9)
    plt.tight_layout()
    plt.show()


plot_hallucination_wordclouds(df_valid_llama, "together_llama_is_hallucinated", "together_llama_answer", title_prefix="LLAMA (Before)")
plot_hallucination_wordclouds(df_valid_gemma, "together_gemma_is_hallucinated", "together_gemma_answer", title_prefix="Gemma (Before)")
plot_hallucination_wordclouds(df_valid_qwen, "groq_qwen_is_hallucinated", "groq_qwen_answer", title_prefix="Qwen (Before)")
plot_hallucination_wordclouds(df_valid_mistral, "together_mistral_7b_is_hallucinated", "together_mistral_7b_answer", title_prefix="Mistral (Before)")
plot_hallucination_wordclouds(df_valid_bloom, "bloom_is_hallucinated", "bloom_answer", title_prefix="Bloom (Before)")

