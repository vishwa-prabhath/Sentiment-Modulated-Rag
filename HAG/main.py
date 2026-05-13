from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load tokenizer and model (GPT2 for generation)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# Load emotion classifier
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
def dynamic_hyperparameter_selector(task_type: str):
    if task_type == "factual":
        return {"temperature": 0.3, "top_p": 0.85, "top_k": 40, "repetition_penalty": 1.2}
    elif task_type == "creative":
        return {"temperature": 0.95, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.0}
    elif task_type == "emotional":
        return {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1}
    else:
        return {"temperature": 0.5, "top_p": 0.85, "top_k": 60, "repetition_penalty": 1.1}

def infer_task_type(prompt):
    result = classifier(prompt)[0]['label']
    if result in ["joy", "surprise"]:
        return "creative"
    elif result in ["sadness", "fear", "love"]:
        return "emotional"
    else:
        return "factual"

def generate_response(prompt: str):
    task_type = infer_task_type(prompt)
    hyperparams = dynamic_hyperparameter_selector(task_type)
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        do_sample=True,
        temperature=hyperparams["temperature"],
        top_p=hyperparams["top_p"],
        top_k=hyperparams["top_k"],
        repetition_penalty=hyperparams["repetition_penalty"],
        max_length=200,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True), task_type

if __name__ == "__main__":
    prompt = input("Enter your prompt: ")
    response, task_type = generate_response(prompt)
    print(f"\n[Task Type: {task_type}]\n")
    print(response)
