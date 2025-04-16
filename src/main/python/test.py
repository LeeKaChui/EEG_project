from llama_cpp import Llama
import time

MODEL_PATH = r"C:\Users\user\Documents\EEG_Project\model\unsloth.Q4_K_M.gguf"

print("Testing with all GPU layers...")
try:
    # First test - attempt to use GPU
    start_load = time.time()
    model = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # Use all layers on GPU
        n_batch=512,
        verbose=True
    )
    end_load = time.time()
    print(f"Model loading time: {end_load - start_load:.2f} seconds")
    
    # Test inference speed
    prompt = "What is 2+2? Answer with just the number."
    start_infer = time.time()
    output = model(prompt, max_tokens=5)
    end_infer = time.time()
    
    response = output["choices"][0]["text"].strip()
    print(f"Response: {response}")
    print(f"Inference time: {end_infer - start_infer:.2f} seconds")
    
except Exception as e:
    print(f"Error with GPU approach: {e}")

print("\nTesting with CPU only for comparison...")
try:
    # Second test - CPU only for comparison
    start_load = time.time()
    model_cpu = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=0,  # Force CPU
        n_batch=512,
        verbose=True
    )
    end_load = time.time()
    print(f"CPU model loading time: {end_load - start_load:.2f} seconds")
    
    # Test inference speed
    prompt = "What is 2+2? Answer with just the number."
    start_infer = time.time()
    output = model_cpu(prompt, max_tokens=5)
    end_infer = time.time()
    
    response = output["choices"][0]["text"].strip()
    print(f"CPU Response: {response}")
    print(f"CPU Inference time: {end_infer - start_infer:.2f} seconds")
    
except Exception as e:
    print(f"Error with CPU approach: {e}")