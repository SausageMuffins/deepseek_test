import os
import time
import json
import re
from llama_cpp import Llama

###########################
# Helper Functions & Setup
###########################

def load_gsm8k_dataset(filepath="gsm8k.jsonl"):
    """
    Load GSM8K-like dataset from a JSONL file.
    Each line is expected to be a JSON object with keys "question" and "answer".
    """
    if not os.path.exists(filepath):
        print(f"GSM8K dataset file '{filepath}' not found. Skipping GSM8K benchmark.")
        return []
    
    dataset = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if "question" in item and "answer" in item:
                    dataset.append(item)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(dataset)} GSM8K benchmark samples.")
    return dataset

def extract_final_number(text):
    """
    Attempt to extract the final number from the text.
    This is a naive approach: it finds the last number in the text.
    """
    # Remove commas in numbers then search for an integer or float.
    text = text.replace(",", "")
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers:
        return numbers[-1]
    return None

def score_gsm8k_sample(prediction, ground_truth):
    """
    Compare the extracted number from prediction with the ground truth.
    Both the prediction and ground truth are processed with extract_final_number.
    Returns True if they match (exact match), else False.
    """
    pred_num = extract_final_number(prediction)
    gt_num = extract_final_number(ground_truth)
    try:
        pred_val = float(pred_num) if pred_num is not None else None
        gt_val = float(gt_num) if gt_num is not None else None
    except ValueError:
        # If conversion fails, do a simple string comparison (fallback)
        return str(prediction).strip() == str(ground_truth).strip()
    
    # Here we require an exact match.
    return pred_val == gt_val

def postprocess_final_answer(output_text):
    """
    If the output contains the marker "Final Answer:", return only what follows.
    Otherwise, return the original output.
    """
    marker = "Final Answer:"
    if marker in output_text:
        # Split on marker and return the text after it (strip leading/trailing whitespace)
        return output_text.split(marker, 1)[-1].strip()
    return output_text

###########################
# Model Wrapper Class
###########################

class GGUFModel:
    """
    A simple wrapper around llama-cpp-python's Llama model.
    """
    def __init__(self, model_path, n_ctx=2048):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.model = None

    def load(self):
        load_start = time.time()
        # If your model supports GPU offloading, adjust n_gpu_layers as needed.
        self.model = Llama(model_path=self.model_path, n_ctx=self.n_ctx, n_gpu_layers=50)
        load_time = time.time() - load_start
        return load_time

    def generate(self, prompt, max_tokens=2048, temperature=0, echo=False, stop=None):
        """
        Generate output text for a given prompt.
        Optionally, a stop sequence can be provided.
        """
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=echo,
            stop=stop  # This parameter is supported in newer versions.
        )
        # llama-cpp-python returns a dict with "choices"
        generated_text = output["choices"][0]["text"] if output.get("choices") else ""
        return generated_text.strip()

###########################
# Benchmark Runner
###########################

def run_inline_benchmarks(model_wrapper):
    """
    Run a few in-line benchmark prompts and return their outputs and timing.
    """
    # Updated prompts: note the "Final Answer:" marker for the complex math prompt.
    benchmarks = {
        "simple_math": "What is 9 + 10?",
        "complex_math": (
            "Solve the following math problem step by step and provide a final answer at the end.\n"
            "What is the integral of x^2 with respect to x?\n"
            "Final Answer:"
        ),
        "general_knowledge": "Tell me a brief history of the Roman Empire."
    }
    
    results = {}
    for name, prompt in benchmarks.items():
        system_instructions = ("Please provide the final answer in your response. "
                               "If the question is simple, don't overthink it and give your answer as 'Final Answer:'."
                               "If it's complex, show your work. Your final answer should be indicated as 'Final Answer:'.\n")
        print(f"\n[Inline] Running benchmark '{name}'")
        print(f"Prompt: {prompt}")
        start = time.time()
        try:
            # Prepend system instructions and add a stop token if desired.
            full_prompt = system_instructions + "\n" + prompt
            output = model_wrapper.generate(full_prompt, max_tokens=2048, temperature=0, stop=["\n\n"])
            # Postprocess to extract only the final answer.
            output = postprocess_final_answer(output)
        except Exception as e:
            output = f"Error: {str(e)}"
        duration = time.time() - start
        print(f"Output: {output}")
        print(f"Generation Time: {duration:.2f} sec")
        results[name] = {"prompt": prompt, "output": output, "generation_time": duration}
    return results

def run_gsm8k_benchmark(model_wrapper, dataset):
    """
    Run the GSM8K benchmark on the provided dataset.
    For each sample, generate an answer and compare with ground truth.
    Returns accuracy and detailed results.
    """
    if not dataset:
        print("No GSM8K samples to run.")
        return {"accuracy": None, "details": []}
    
    correct = 0
    details = []
    for idx, sample in enumerate(dataset):
        if idx == 100: # Limit to 100 samples for now
            break
        question = sample["question"]
        gt_answer = sample["answer"]
        system_instructions = ("Please provide the final answer in your response. "
                               "If the question is simple, don't overthink it and give your answer as 'Final Answer:'."
                               "If it's complex, show your work. Your final answer should be indicated as 'Final Answer:'.\n\n")
        
        prompt = system_instructions + question
        print(f"\n[GSM8K] Sample {idx+1}:")
        print(f"Question: {question}")
        start = time.time()
        try:
            prediction = model_wrapper.generate(prompt, max_tokens=2048, temperature=0)
        except Exception as e:
            prediction = f"Error: {str(e)}"
        duration = time.time() - start
        is_correct = score_gsm8k_sample(prediction, gt_answer)
        if is_correct:
            correct += 1
        details.append({
            "question": question,
            "ground_truth": gt_answer,
            "prediction": prediction,
            "is_correct": is_correct,
            "generation_time": duration
        })
        print(f"Prediction: {prediction}")
        print(f"Correct: {is_correct} (Time: {duration:.2f} sec)")
    
    accuracy = correct / len(dataset)
    print(f"\n[GSM8K] Accuracy: {accuracy*100:.2f}% ({correct} out of {len(dataset)})")
    return {"accuracy": accuracy, "details": details}

###########################
# Main Script
###########################

def main():
    MODELS_DIR = "models"
    results = {}
    
    # Load GSM8K dataset (if available)
    gsm8k_dataset = load_gsm8k_dataset("gsm8k.jsonl")
    
    # List all gguf files in the models directory
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
    if not model_files:
        print("No gguf models found in the 'models' directory.")
        return
    
    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        print(f"\n========== Benchmarking Model: {model_file} ==========")
        
        # Initialize the model wrapper
        model_wrapper = GGUFModel(model_path=model_path, n_ctx=2048)
        try:
            load_time = model_wrapper.load()
            print(f"Model loaded in {load_time:.2f} seconds.")
        except Exception as e:
            print(f"Failed to load model {model_file}: {e}")
            results[model_file] = {"error": str(e)}
            continue
        
        model_results = {"load_time": load_time}
        
        # # Run inline benchmarks
        # inline_results = run_inline_benchmarks(model_wrapper)
        # model_results["inline_benchmarks"] = inline_results
        
        # Run GSM8K benchmark if dataset available
        gsm8k_results = run_gsm8k_benchmark(model_wrapper, gsm8k_dataset)
        model_results["gsm8k_benchmark"] = gsm8k_results
        
        results[model_file] = model_results
        
    # Save all results to a JSON file
    with open("benchmark_results.json", "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=2)
    print("\n========== Benchmarking Completed ==========")
    print("Results saved to 'benchmark_results.json'.")

if __name__ == "__main__":
    main()
