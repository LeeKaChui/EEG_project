from flask import Flask, request, jsonify
from llama_cpp import Llama
import os

app = Flask(__name__)

# Load the model once when the app starts
model_path = r"C:\Users\user\Documents\EEG_Project\model\unsloth.Q4_K_M.gguf"
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_gpu_layers=-1
)

@app.route('/classify', methods=['POST'])
def classify_eeg():
    try:
        data = request.json
        if not data or 'eeg_data' not in data:
            return jsonify({"error": "No EEG data provided"}), 400
        
        eeg_data = data.get('eeg_data')
        
        prompt = f"""
### Task: Classify the following EEG data type into 0 or 1.
### Input:
{eeg_data}
### Response:
"""
        
        output = llm(
            prompt,
            max_tokens=5,
            temperature=0.1,
            stop=["###"]
        )
        
        response_text = output["choices"][0]["text"].strip()
        
        # Try to get a clean 0 or 1 from the response
        result = None
        for char in response_text:
            if char in ['0', '1']:
                result = int(char)
                break
        
        return jsonify({
            "result": result,
            "raw_response": response_text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Simple health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    # Set host to '0.0.0.0' to make it accessible from other devices on the network
    app.run(debug=True, host='0.0.0.0', port=5000)