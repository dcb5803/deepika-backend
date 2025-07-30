from flask import Flask, request, jsonify
from flask_cors import CORS
from ctransformers import AutoModelForCausalLM

app = Flask(__name__)
CORS(app)

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",  # Make sure this file is in the same folder
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
