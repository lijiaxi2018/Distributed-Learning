from deepsparse import TextGeneration

def run_deepsparse(model_name="zoo:llama2-7b-gsm8k_llama2_pretrain-pruned60_quantized"):
    pipeline = TextGeneration(model=model_name)

    prompt="""
    Please answer the following math problem. Problem: Find all x that satisfy the inequality (2x+10)(x+3) < (3x+9)(x+8). Express your answer in interval notation.
    """
    print(pipeline(prompt, max_new_tokens=200).generations[0].text)
