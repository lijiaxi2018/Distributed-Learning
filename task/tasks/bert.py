from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

def run_bert(model_name="camille/bert-base-pruned-voc-esw0.9-40000-en-de-cased", file_name="/home/jiaxi/cs525/Distributed-Learning/data/tinyshakespeare-2000.txt"):
    pipe = pipeline("feature-extraction", model=model_name, device="cuda")
    dataset = load_dataset('text', data_files=file_name)

    for out in tqdm(pipe(KeyDataset(dataset['train'], "text"))):
        # print(out)
        pass