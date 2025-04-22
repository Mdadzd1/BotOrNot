import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

# Initialize model and tokenizer to None
model = None
tokenizer = None

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
local_model_path = os.path.join(script_dir, "trained_bot_detector")

try:
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    st.success("Model and tokenizer loaded successfully from local directory.")
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
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
            ai_probability = probabilities[0, 1].item()
        return ai_probability
    else:
        # Placeholder behavior if the model couldn't load
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
st.markdown("""
This section outlines how your team can regularly train the AI detection model.

**Phase 1: Data Collection & Preparation**
1. **Regularly collect new examples** of both human-written and AI-generated text. Store this data in a structured format (e.g., CSV, JSON files, or a dedicated data storage).
2. **Preprocess the data:** This might involve cleaning, tokenization, and formatting it for your chosen machine learning framework (e.g., PyTorch, TensorFlow).
3. **Split the data** into training, validation, and testing sets.

**Phase 2: Model Training & Evaluation (To be run offline or in a dedicated training environment)**
1. **Choose a model architecture:** Select a suitable transformer model (e.g., DistilBERT, RoBERTa) for text classification.
2. **Implement a training script:** Use a framework like Hugging Face Transformers to fine-tune the model on your prepared data. This script will:
   - Load the pre-trained model and tokenizer.
   - Load your training and validation datasets.
   - Define training parameters (learning rate, epochs, batch size, etc.).
   - Train the model.
   - Evaluate the model on the validation set.
   - Save the trained model weights and configuration to the `trained_bot_detector` directory.
3. **Track training progress:** Use tools like TensorBoard or Weights & Biases to monitor metrics and compare different training runs.

**Phase 3: Model Deployment & Integration**
1. **Save the trained model:** After achieving satisfactory performance on the validation set, ensure the final trained model files are in the `trained_bot_detector` directory.
2. **Update the Streamlit app:** The current `BotOrNot.py` is already set up to load from this directory. Ensure the `trained_bot_detector` directory is in the same directory as `BotOrNot.py` in your GitHub repository.
3. **Deploy the updated app:** Push the changes to your GitHub repository, and Streamlit Cloud will automatically update the deployed app.

**Regular Training Cycle:**
1. **On a scheduled basis (e.g., weekly, monthly), repeat Phase 1 and Phase 2.**
2. **After training and evaluation, if the new model shows improved performance, ensure the `trained_bot_detector` directory in your GitHub repository is updated with the new model files.** Streamlit Cloud will use these upon the next deployment or restart.

**Collaboration:**
- Use a version control system like Git and platforms like GitHub for code sharing and collaboration.
- Establish clear roles and responsibilities within your team for data collection, training, and deployment.
- Use branches in Git to manage different versions of the code and model during development and training.
"""
)

import random # Import the random module here