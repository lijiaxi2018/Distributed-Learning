import time
import power.AGXPowerLogger as APL
from dvfs.lib import setCpu, setGpu, getCpuStatus, getGpuStatus
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_llama2(model_name="neuralmagic/Llama-2-7b-ultrachat"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    input_text = "Write me a poem about Machine Learning."
    # input_ids = tokenizer.apply_chat_template(input_text, add_generation_prompt=True, return_tensors="pt").to("cuda")
    input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=True).to("cuda")

    outputs = model.generate(
        **input_ids,
        max_length=50,
        min_length=40,
        # num_return_sequences=1,
        # temperature=0.7,
    )
    print(tokenizer.decode(outputs[0]))

def run_llama2_dataset(model_name="neuralmagic/Llama-2-7b-pruned50-retrained-ultrachat", input_file_path="/home/jiaxi/cs525/Distributed-Learning/data/llama2_text.txt"):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # Load the dataset from the input text file
    dataset = load_dataset('text', data_files=input_file_path)['train']

    # Iterate over each line in the dataset
    for example in dataset:
        input_text = example['text']  # Assuming each line in your txt file is a separate example

        # Tokenize the input text
        input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=True).to("cuda")

        # Generate output
        outputs = model.generate(
            **input_ids,
            max_length=50,
            min_length=40,
        )

        # Print the decoded output
        # print(tokenizer.decode(outputs[0]))