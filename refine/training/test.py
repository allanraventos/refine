from vllm import LLM, SamplingParams
import argparse
import datasets
import pickle
from transformers import AutoTokenizer
import os
import json
import torch
from datasets import load_from_disk

# Get the number of GPUs available
num_gpus = torch.cuda.device_count()

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--T", type=float, default=1)
parser.add_argument("--N", type=int, default=1)
parser.add_argument("--bfs", action="store_true")
parser.add_argument(
    "--dataset_name",
    type=str,
    default="openai/gsm8k",
    help="The name of the dataset to use (via the datasets library).",
)
parser.add_argument(
    "--training_set",
    action="store_true",
)
parser.add_argument(
    "--dataset_config_name",
    type=str,
    default="main",
    help="The configuration name of the dataset to use (via the datasets library).",
)
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--prompt_template", type=str)
parser.add_argument("--test", action="store_true")

args = parser.parse_args()


prompt_gsm8k = "Question: {qe}\nAnswer: {ans}"
prompt_math = "Question: {qe}\nAnswer: {ans}"

base_format = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: {qe}\n\n### Response: Let's think step by step. {ans}"

llama_format = """<|start_header_id|>system<|end_header_id|>
You're a helpful assistant that answers math problems with steps.<|eot_id|><|start_header_id|>user<|end_header_id|>
{qe}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{ans}"""

prompt_registry = {
    "gsm8k": prompt_gsm8k,
    "math": prompt_math,
    "llama_instruct": llama_format,
    "base_format": base_format,
}


prompt = prompt_registry[args.prompt_template]

prefix_column = "question"
completion_column = "answer"

if "gsm8k" in args.dataset_name:
    if args.training_set:
        test_data = datasets.load_dataset(
            args.dataset_name, args.dataset_config_name, split="train"
        )
    else:
        test_data = datasets.load_dataset(
            args.dataset_name, args.dataset_config_name, split="test"
        )
elif "math" in args.dataset_name:
    if args.training_set:
        test_data = load_from_disk(args.dataset_name)["train"]
    else:
        test_data = load_from_disk(args.dataset_name)["test"]
test_list = [prompt.format(qe=ex[prefix_column], ans="") for ex in test_data]

if args.test:
    test_list = test_list[430:440]

repititions = 1
NN = args.N
if args.N >= 256:
    assert args.N % 256 == 0, "N must be a multiple of 256 for tensor parallelism"
    repititions = args.N // 256
    args.N = 256

if args.greedy:
    print("Using greedy decoding")
    args.N = 1
    repititions = 1
    sampling_params = SamplingParams(
        temperature=0, n=args.N, logprobs=1, max_tokens=args.max_length
    )
elif args.bfs:
    print("Using BFS decoding")
    sampling_params = SamplingParams(
        temperature=0,
        n=args.N,
        best_of=args.N,
        logprobs=1,
        max_tokens=args.max_length,
        use_beam_search=True,
    )
else:
    sampling_params = SamplingParams(
        temperature=args.T, n=args.N, logprobs=1, max_tokens=args.max_length
    )

model = LLM(
    model=args.model, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.98
)

for reptition in range(repititions):

    vllm_outputs = model.generate(test_list, sampling_params)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    results = []
    for vllm_output in vllm_outputs:
        results.append(
            [
                (tokenizer.decode(output.token_ids), output.cumulative_logprob)
                for output in vllm_output.outputs
            ]
        )
    training_set = "train" if args.training_set else ""
    if args.test:
        save_dir = os.path.join(args.model, f"results_test.pkl")
    else:
        save_dir = os.path.join(
            args.model, f"results_{NN}_{args.T}_{reptition}_{training_set}.pkl"
        )
        if 'AIME24' in args.dataset_name:
            save_dir = os.path.join(args.model, f"results_AIME24_{NN}_{args.T}_{reptition}_{training_set}.pkl")
        if args.greedy:
            save_dir = os.path.join(args.model, f"results_greedy_{training_set}.pkl")
            if 'AIME24' in args.dataset_name:
                save_dir = os.path.join(args.model, f"results_greedy_AIME24_{training_set}.pkl")
        elif args.bfs:
            save_dir = os.path.join(args.model, f"results_bfs_{NN}_{training_set}.pkl")
            if 'AIME24' in args.dataset_name:
                save_dir = os.path.join(args.model, f"results_bfs_AIME24_{NN}_{training_set}.pkl")
    with open(save_dir, "wb") as f:
        pickle.dump(results, f)
