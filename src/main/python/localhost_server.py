from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from llama_cpp import Llama
import os
import time
import json
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'eeg_classifier_secret_key'

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

session_storage = {}

llm = None

try:
    model_path = r"C:\Users\user\Documents\EEG_Project\model\unsloth.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        print(f"Warning: Model file does not exist at {model_path}")
    else:
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1
        )
        print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")

current_dir = os.path.dirname(os.path.abspath(__file__))
js_dir = os.path.join(os.path.dirname(current_dir), "Js")

@app.route('/')
def index():
    return send_from_directory(js_dir, 'localhost_client_web.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/classify', methods=['POST'])
def classify_eeg():
    if llm is None:
        return jsonify({
            "error": "Model not loaded correctly. Check server logs."
        }), 500
    
    try:
        data = request.json
        if not data or 'eeg_data' not in data:
            return jsonify({"error": "No EEG data provided"}), 400
        
        eeg_data = data.get('eeg_data')
        
        session_id = request.cookies.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
        
        results = []
        
        for _ in range(10):
            prompt = f"""
### Task: Classify the following EEG data as EITHER "0: Alcohol use disorder" OR "1: Depressive disorder". 
Both classes are equally likely - carefully examine the features to determine which is correct.
### Input:
{eeg_data}
### Response:
"""
            
            time.sleep(0.1)
            
            output = llm(
                prompt,
                max_tokens=5,
                temperature=0.1,
                stop=["###"]
            )
            
            response_text = output["choices"][0]["text"].strip()
            
            result = None
            for char in response_text:
                if char in ['0', '1']:
                    result = int(char)
                    break
            
            if result is not None:
                results.append(result)
        
        if not results:
            probability = 0
            final_result = None
        else:
            ones_count = results.count(1)
            probability = (ones_count / len(results)) * 100
            
            final_result = 1 if ones_count > len(results) / 2 else 0
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "eeg_data": eeg_data,
            "results": results,
            "probability": probability,
            "final_result": final_result
        }
        
        if session_id not in session_storage:
            session_storage[session_id] = []
        
        session_storage[session_id] = [history_entry] + session_storage[session_id]
        
        if len(session_storage[session_id]) > 20:
            session_storage[session_id] = session_storage[session_id][:20]
        
        response = jsonify({
            "result": final_result,
            "probability": probability,
            "individual_results": results,
            "session_id": session_id
        })
        
        response.set_cookie('session_id', session_id, max_age=86400, samesite='None', secure=False, path='/')
        
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    session_id = request.cookies.get('session_id')
    if not session_id or session_id not in session_storage:
        return jsonify({"history": []})
    
    return jsonify({"history": session_storage[session_id]})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session_id = request.cookies.get('session_id')
    if session_id and session_id in session_storage:
        session_storage[session_id] = []
    
    return jsonify({"status": "success", "message": "History cleared"})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "model_loaded": llm is not None
    })

if __name__ == '__main__':
    print(f"Server starting... HTML file should be at: {os.path.join(current_dir, 'localhost_client_web.html')}")
    print(f"Check if file exists: {os.path.exists(os.path.join(current_dir, 'localhost_client_web.html'))}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
