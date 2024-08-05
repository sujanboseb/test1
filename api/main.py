import re
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Define the custom time labeling functions
def label_times(text):
    time_pattern = r'(\b\d{1,2}:\d{2}\s?[ap]\.?m\.?\b|\b\d{1,2}\s?[ap]\.?m\.?\b|\b\d{1,2}:\d{2}\b)'
    time_matches = list(re.finditer(time_pattern, text, re.IGNORECASE))
    
    new_ents = []

    if len(time_matches) > 0:
        start_time_match = time_matches[0]
        new_ents.append((start_time_match.start(), start_time_match.end(), "STARTING_TIME"))

    if len(time_matches) > 1:
        end_time_match = time_matches[1]
        # Check if the starting and ending times are the same
        if start_time_match.group() == end_time_match.group():
            new_ents.append((end_time_match.start(), end_time_match.end(), "ERROR"))
        else:
            new_ents.append((end_time_match.start(), end_time_match.end(), "ENDING_TIME"))

    if len(time_matches) > 2:
        for match in time_matches[2:]:
            new_ents.append((match.start(), match.end(), "ERROR"))

    return new_ents

def format_output(text, entities):
    formatted_output = ""
    for start, end, label in entities:
        if label == "STARTING_TIME":
            formatted_output += f"starting_time: {text[start:end]} "
        elif label == "ENDING_TIME":
            formatted_output += f"ending_time: {text[start:end]} "
        elif label == "ERROR":
            formatted_output += f"error: {text[start:end]} "
    return formatted_output.strip()

# Load the tokenizer and model
tokenizer = BartTokenizer.from_pretrained("tokenizer")
model = BartForConditionalGeneration.from_pretrained("model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the FastAPI app
app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict/")
async def predict(input_text: InputText):
    text = input_text.text
    
    # Process text with the BART model
    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True, padding='max_length')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    summary_ids = model.generate(inputs['input_ids'], max_length=1024, num_beams=4, early_stopping=True)
    predicted_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Label time expressions
    entities = label_times(text)
    formatted_output = format_output(text, entities)

    return {"prediction": predicted_text, "entities": formatted_output}

# Endpoint for handling WhatsApp webhook
@app.post("/webhook/")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    message = data['messages'][0]['text']['body'] if 'messages' in data and data['messages'] else None

    if message:
        input_text = InputText(text=message)
        prediction = await predict(input_text)
        response_text = f"Prediction: {prediction['prediction']}\nEntities: {prediction['entities']}"
    else:
        response_text = "No message received."

    return {"text": response_text}
