from refine.training.loss_functions import (
    custom_loss_function2,
)
from refine.training.NCO import custom_loss_function1
from refine.training.filter_samples import (
    filter_fn_NCO,
    non_filter,
    filter_fn_vllm,
    filter_fn_NCO_formal,
)

prompt_gsm8k = "Question: {qe}\nAnswer: {ans}"
prompt_math = "Question: {qe}\nAnswer: {ans}"

prompt_math_no_COT = "Please give the answer to the following question directly! No need to write any steps.\nQuestion: {qe}\nAnswer: {ans}"

loss_registry = {
    "custom_loss_function1": custom_loss_function1,
    "custom_loss_function2": custom_loss_function2,
}
base_format = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: {qe}\n\n### Response: Let's think step by step. {ans}"
llama_format = """<|start_header_id|>system<|end_header_id|>
You're a helpful assistant that answers math problems with steps.<|eot_id|><|start_header_id|>user<|end_header_id|>
{qe}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{ans}"""

formal_math = "{qe}:::{ans}"

prompt_registry = {
    "gsm8k": prompt_gsm8k,
    "math": prompt_math,
    "math_no_COT": prompt_math_no_COT,
    "llama_instruct": llama_format,
    "base_format": base_format,
    "formal_math": formal_math,
}

filter_registry = {
    "NCO": filter_fn_NCO,
    "non_filter": non_filter,
    "cot": filter_fn_vllm,
    "NCO_formal": filter_fn_NCO_formal,
}
