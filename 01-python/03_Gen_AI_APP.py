import os
from dotenv import load_dotenv
import boto3

load_dotenv() # .env file content into the program
client = boto3.client('bedrock-runtime')
model_id = 'mistral.mistral-7b-instruct-v0:2'

while True:
    user_input = input("prompt: ").strip().lower()

    if user_input == "exit":
        print("exiting")
        break
    else:
        response = client.converse(
            modelId = model_id,
            messages = [
                {
                    "role": "user",
                    "content": [{"text": user_input}]
                }
            ],
            inferenceConfig = {"temperature": 0.7, "maxTokens": 100}
        )

        print(response["output"]["message"]["content"][0]["text"])