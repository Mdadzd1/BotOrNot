import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import random

# Initialize model and tokenizer to None
model = None
tokenizer = None
fallback_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
fallback_model = None
fallback_tokenizer = None

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
local_model_path = os.path.join(script_dir, "trained_bot_detector")

try:
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    st.success("Successfully loaded your trained model.")
except Exception as e:
    st.error(f"Error loading your trained model: {e}")
    st.info(f"Falling back to the '{fallback_model_name}' model.")
    try:
        fallback_model = AutoModelForSequenceClassification.from_pretrained(fallback_model_name)
        fallback_tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
        model = fallback_model
        tokenizer = fallback_tokenizer
        st.success(f"Successfully loaded the fallback model: '{fallback_model_name}'.")
    except Exception as fallback_e:
        st.error(f"Error loading fallback model '{fallback_model_name}': {fallback_e}")
        st.info("The app will run in an untrained, placeholder mode.")

device = torch.device("cpu")
if model is not None:
    model.to(device)
    model.eval()

st.title("🤖 Bot or Not?")
st.subheader("Check if a piece of text is likely AI-generated")

def predict_ai_generated(text):
    if model is not None and tokenizer is not None:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

            # Assuming a binary classification model (check the loaded model's output)
            # For sentiment analysis, index 1 might be "positive" - adjust accordingly
            if fallback_model_name in model.config.name_or_path:
                # Adjust for sentiment analysis (positive might be more "human-like"?)
                ai_probability = 1 - probabilities[0, 1].item()
            else:
                ai_probability = probabilities[0, 1].item() # Assuming index 1 is "AI-like"
            return ai_probability
    else:
        st.warning("Running in untrained, placeholder mode.")
        return random.random()

user_input = st.text_area("Paste your text here", height=200)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        ai_probability = predict_ai_generated(user_input)
        bot_score = round(ai_probability * 100, 2)
        st.success(f"🤖 AI Probability: **{bot_score}%**")
        if bot_score > 50:
            st.info("This text is likely AI-generated.")
        else:
            st.info("This text is likely human-written.")

st.markdown("---")
st.subheader("Team Training Workflow")
# ... (rest of the workflow description remains the same)

import random