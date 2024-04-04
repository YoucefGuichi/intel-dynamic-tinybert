from fastapi import FastAPI
from transformers import pipeline
import uvicorn
import os 

app = FastAPI()

# Load the question answering model
model = pipeline("question-answering", model="Intel/dynamic_tinybert")

@app.post("/intel-dynamic-tinybert/")
async def question_answering(question_context: dict):
    question = question_context.get("question")
    context = question_context.get("context")
    
    # Use the model to get the answer
    answer = model(question, context)
    return {
        "answer": answer,
        }
