import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from ctransformers import AutoModelForCausalLM

MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH = "tinyllama.gguf"

# Auto-download model if not exists
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model...")
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("✅ Model downloaded!")

app = Flask(__name__)
CORS(app)

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    model_file=MODEL_PATH,
    model_type="llama",
    max_new_tokens=256,
    temperature=0.7
)

custom_data = {
    "hi": "hello Good morning, I am chatbot Deepika",
    "about you": "I am Deepika AI Chatbot, how may I help you?",
    "i love you": "I love you too",
    "age": "45"
}

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    msg = data.get("message", "")
    if msg in custom_data:
        return jsonify({"response": custom_data[msg]})
    result = model(f"### Instruction:\n{msg}\n\n### Response:\n")
    return jsonify({"response": result})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)

