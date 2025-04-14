import functools
import multiprocessing as mp
import re
import signal
import time

import numpy as np
import ray
import sympy
import torch
import torch.distributed as dist
from sympy.parsing.latex import parse_latex
from tqdm import tqdm

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    r"\left",
    r"\right",
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = re.sub(r"\\text{(.*?)}", r"\1", final_answer)
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def process_results(results: str):
    candidates = results.replace("</s>", "")
    answer = normalize_final_answer(candidates)
    return answer


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    # Example failure of this code: "\fbox[t" != "\boxed{"
    # assert s[: len(left)] == left, f"{s[: len(left)]} != {left}"
    # assert s[-1] == "}"

    # Just return a dummy thing if the assertion fails (FIXME)
    if s[: len(left)] != left:
        return s
    elif s[-1] != "}":
        return s

    return s[len(left) : -1]


def last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


# TODO : add support for beam_search and greedy
# TODO : add support for batch_size larger than 1
def is_answer_correct(model_answer, correct_answer):
    last_boxed_string = last_boxed_only_string(model_answer)
    if last_boxed_string is None:
        # print("No boxed string found")
        return False

    filtered_result = remove_boxed(last_boxed_string)

    # NOTE: here no `correct_answer[0]` (assuming correct answer is a single string)
    return normalize_final_answer(filtered_result) == normalize_final_answer(
        correct_answer
    )

    # Previous logic
    # if filtered_result.replace(",", "").replace(" ", "") == correct_answer[0].replace(
    #     ",", ""
    # ).replace(" ", ""):
    #     return True
    # else:
    #     return False


def check_one(args):
    """Compute equivalences for the i-th element in parallel."""
    i, results_i, ans_str = args  # Unpack
    # results_i: list of length n, each sub-item might be a tuple [("some latex expression", ...), ...]
    # ans_str: the single "correct" answer for the i-th item

    c_list = []

    # Attempt to parse the "gold" answer
    try:
        # ans_processed = process_results(ans_str)
        ans_processed = ans_str

        ans_parsed = parse_latex(ans_processed)
    except (
        sympy.parsing.latex.errors.LaTeXParsingError,
        sympy.SympifyError,
        TypeError,
    ):
        print(f"Failed to parse:")
        print(f"{ans_str} ->")
        print(f"{ans_processed} ->")
        print("-" * 80)
        ans_parsed = None

    for r_tuple in results_i:
        # Suppose r_tuple is something like ( "latex_string", ...)
        # r_str = r_tuple[0]  # FIXME: don't think this is a tuple
        r_str = r_tuple
        # Pre-process
        # candidate_processed = process_results(r_str)
        candidate_processed = r_str

        # Try parsing
        try:
            candidate_parsed = parse_latex(candidate_processed)

            # print(f"candidate: {candidate_parsed}, answer: {ans_parsed}")

            # If ans_parsed is None, we fallback to string-equality
            if ans_parsed is not None:
                # c_list.append(is_equiv(str(candidate_parsed), str(ans_parsed)))
                c_list.append(is_equiv(candidate_parsed, ans_parsed))
            else:
                # If we couldn't parse the gold, do something fallback
                c_list.append(candidate_processed == ans_str)
        except (
            sympy.parsing.latex.errors.LaTeXParsingError,
            sympy.SympifyError,
            TypeError,
        ):
            # print(f"Failed to parse {candidate_processed}")
            # Parsing failure => fallback check
            if ans_parsed is None:
                c_list.append(candidate_processed == ans_str)
            else:
                c_list.append(False)

    # print("Completing list in one process")

    # You can do any final logic here, like storing c_list somewhere or returning
    return c_list


def parallel_sympy_equivalence_check(results, answers, n_processes=4):
    """
    :param results: list of length bs_per_engine; each element is a list of length n
    :param answers: list of length bs_per_engine
    :param n_processes: number of processes to use
    :return: list of length bs_per_engine, each item is c_list of booleans
    """
    # Prepare data for parallel mapping
    # Each item: (i, results[i], answers[i])
    tasks = [(i, results[i], answers[i]) for i in range(len(answers))]

    # Use multiprocessing Pool
    with mp.Pool(n_processes) as pool:
        # If you want a progress bar, use imap or imap_unordered + tqdm
        out = list(tqdm(pool.imap(check_one, tasks), total=len(tasks)))

    return out


"""
This should very likely just be multiprocessed (should work fine)
"""


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


import sympy
from sympy import Eq, Matrix
from sympy.sets.sets import Set
from sympy.core.relational import Relational


def is_equiv(x1, x2):
    """Return True if x1 and x2 are 'equivalent', handling sets, matrices, equations, etc."""
    # 1) If they are exactly the same type, we can attempt .equals().
    #    But for many objects, 'equals()' is the right approach.
    #    We just wrap it in a try/except because .equals() can raise for some sympy types.
    try:
        # Sympy often returns True/False, but can also return a Sympy expression.
        # Check if it is the literal Python True:
        eq_result = x1.equals(x2)
        if eq_result is True:
            return True
    except Exception:
        pass

    # 2) If both are sets (intervals, unions, etc.), we can do a set comparison:
    if isinstance(x1, Set) and isinstance(x2, Set):
        # For sets, sympy often allows direct == or .equals().
        # If .equals() did not return True above, try direct equality:
        _ret = x1 == x2
        if not _ret:
            print(f"Failed set comp: {x1} and {x2}")
        return _ret
        # return x1 == x2

    # 3) If both are equations/relational (like Eq, Lt, Gt):
    if isinstance(x1, Relational) and isinstance(x2, Relational):
        # For equations, we can attempt eq1.equals(eq2). If that was inconclusive above,
        # you might want to do eq1 - eq2 style checks. But typically eq1.equals(eq2) is best.
        # If parse_latex gave you an Eq(y, 63/32), that is the object to compare with eq2.
        # If .equals(...) wasn’t True up above, we’ll just see if they are structurally identical:
        _ret = x1 == x2
        if not _ret:
            print(f"Failed rel comp: {x1} and {x2}")
        return _ret
        # return x1 == x2

    # 4) If both are matrices, we can subtract them and see if the result is zero:
    if isinstance(x1, Matrix) and isinstance(x2, Matrix):
        if x1.shape != x2.shape:
            return False
        # Check if difference is a zero matrix:
        _ret = (x1 - x2).is_zero
        if not _ret:
            print(f"Failed at mat compare: {x1} and {x2}")
        return _ret
        # return (x1 - x2).is_zero

    # 5) Otherwise, assume they are ordinary expressions; try subtract + simplify:
    try:
        diff = x1 - x2
        if sympy.simplify(diff) == 0:
            return True
        else:
            print(f"Failed at diff/simplify: {x1} and {x2}")
            return False
    except Exception:
        pass

    print(f"Failed somewhere else: {x1} and {x2}")
    return False


# @functools.lru_cache(maxsize=4096)
# def is_equiv(x1: str, x2: str) -> bool:
#     """
#     x1 and x2 are normalized latex string
#     """
#     try:
#         with timeout(seconds=10):

#             parsed_x1 = x1
#             parsed_x2 = x2

#             # e.g. two Eq
#             try:
#                 if parsed_x1.equals(parsed_x2):
#                     return True
#             except TypeError:
#                 # print(f"couldn't subtract {x1} and {x2}")
#                 pass

#             try:
#                 diff = parsed_x1 - parsed_x2
#                 if sympy.simplify(diff) == 0:
#                     return True
#                 else:
#                     return False
#             except ValueError:
#                 print(f"couldn't subtract/simplify {x1} and {x2}")
#                 return False

#     except TimeoutError:
#         return False
#     except ImportError as e:
#         return False


def filter_training_data(
    model, tokenizer, batch, sampling_parameters, number_of_samples, max_length=2048
) -> bool:
    assert len(batch["question"]) == 1

    # Set model to eval for running inference
    model.eval()

    with torch.no_grad():
        for _ in range(number_of_samples):
            result_id = model.generate(
                batch["question"],
                max_length=max_length,
                temperature=1.0,
                do_sample=True,
            )[0]

            result = tokenizer.decode(result_id)

            if is_answer_correct(result, batch["answer"]):
                model.train()
                return True

    # Set model to train prior to continuing training
    model.train()
    return False


# NCO filter

from refine.training.NCO import factor, log_prob


def filter_training_data_NCO(model, batch, N, threshold=0.1) -> bool:
    assert len(batch["question"]) == 1
    # Set model to eval for running inference
    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
        )

    factor_val = factor(outputs.logits, batch["labels"], N).item()

    # ToDo: Add logging for factor
    # print(f"factor_val: {factor_val}")
    # Set model to train prior to continuing training
    model.train()
    return factor_val < threshold


import accelerate


def exact_match(results, answers):
    out = []

    n_samples = len(results)
    N = len(results[0])

    for i in tqdm(range(n_samples)):
        is_correct = [
            is_answer_correct(solution, answers[i]) for solution in results[i]
        ]
        out.append(is_correct)

    return np.array(out)


# @torch.no_grad()
def filter_fn_vllm(
    engine,
    inds,
    dataset,
    tokenizer,
    prompt_format,
    controller_id,  # None if process is not a controller
    n_controller_processes,
    sampling_params,
    accelerator,
    sleep_group,
    threshold=0.2,  # If fewer than this fraction of passes then sample is hard
):

    is_hard_tensor = torch.zeros(len(inds), dtype=bool).to("cuda")
    sample_success_prob_tensor = torch.zeros(len(inds), dtype=float).to("cuda")

    _t0 = time.time()

    if controller_id is not None:

        bs_per_engine = len(inds) // n_controller_processes
        start_i = bs_per_engine * controller_id
        end_i = bs_per_engine * (controller_id + 1)
        sample_list = [
            prompt_format.format(qe=dataset[_i]["question"], ans="") for _i in inds
        ][start_i:end_i]

        # TODO: is this logic in the main script? Should it be?
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # tokenizer.pad_token = tokenizer.eos_token

        # `outputs` is a list of length bs, where entry `outputs[i].outputs` is length
        # N, each `outputs[i].outputs[j]` has a token_ids field.
        outputs = ray.get(engine.generate.remote(sample_list, sampling_params))

        decoded_outputs = [
            [tokenizer.decode(entry.token_ids) for entry in output_group.outputs]
            for output_group in outputs
        ]

        # NOTE: bug fix
        answers = [dataset[_i]["answer"] for _i in inds][start_i:end_i]
        assert len(decoded_outputs) == len(answers)
        exact_match_results = exact_match(decoded_outputs, answers)
        assert exact_match_results.shape == (bs_per_engine, sampling_params.n)

        is_hard = exact_match_results.sum(axis=1) == 0
        sample_success_prob = exact_match_results.sum(axis=1) / sampling_params.n
        is_hard = sample_success_prob < threshold

        is_hard_tensor[start_i:end_i] = torch.tensor(is_hard, dtype=bool).to("cuda")
        sample_success_prob_tensor[start_i:end_i] = torch.tensor(
            sample_success_prob, dtype=float
        ).to("cuda")

    # CPU barrier to allow all processes to do finish doing inference. Otherwise processes that
    # are only workers will "hang" at 100% util without contributing meaningfully to inference.
    dist.barrier(group=sleep_group)

    # NOTE: this reduce point is a problem. It seems to lock the non-controller processes.
    # What is another way to handle this?
    is_hard_tensor = accelerator.reduce(is_hard_tensor, reduction="sum").cpu().numpy()
    sample_success_prob_tensor = (
        accelerator.reduce(sample_success_prob_tensor, reduction="sum").cpu().numpy()
    )

    is_hard = np.where(is_hard_tensor)[0].tolist()
    hard_indices = [inds[_i] for _i in is_hard]
    sample_success_prob = sample_success_prob_tensor[is_hard]

    if accelerator.is_main_process:
        print(
            f"Filter runtime: {time.time() - _t0:.2f} seconds, {len(hard_indices)} hard out of {len(inds)}"
        )

    return (hard_indices, sample_success_prob.tolist())


@torch.no_grad()
def filter_fn_NCO(
    model,
    accelerator,
    inds,
    dataset,
    collate_fn,
    global_batch_size,
    is_main_process,
    N,
    threshold=0.1,
):

    world_size = accelerator.num_processes

    assert global_batch_size % world_size == 0

    # if is_main_process:
    inds = torch.tensor(inds, dtype=int).reshape(world_size, -1).to(accelerator.device)
    # else:
    #     inds = torch.zeros(world_size, global_batch_size // world_size, dtype=int).to(
    #         accelerator.device
    #     )

    factor_val_gather = torch.zeros_like(inds, dtype=float).to(accelerator.device)

    inds_copy = inds.clone()

    # if is_main_process:
    #     import pdb

    #     pdb.set_trace()

    batch = collate_fn(dataset.select(inds[accelerator.local_process_index]))

    # assert len(batch["question"]) == 1
    # Set model to eval for running inference
    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"].to(accelerator.device),
            labels=batch["labels"].to(accelerator.device),
            attention_mask=batch["attention_mask"].to(accelerator.device),
        )

    factor_val = factor(outputs.logits, batch["labels"].to(accelerator.device), N)

    # ToDo: Add logging for factor
    # print(f"factor_val: {factor_val}")
    # Set model to train prior to continuing training

    factor_val_gather[accelerator.local_process_index] = factor_val

    factor_val_gather = accelerator.reduce(factor_val_gather, reduction="sum")

    model.train()

    inds_copy = inds_copy.reshape(-1)[factor_val_gather.reshape(-1) > threshold]

    return (
        inds_copy.cpu().tolist(),
        torch.ones_like(inds_copy).float().tolist(),
    )


@torch.no_grad()
def non_filter(
    **kwargs,
):
    return (
        kwargs["inds"],
        torch.ones(len(kwargs["inds"])).float().tolist(),
    )


from refine.training.NCO import f as fff


@torch.no_grad()
def filter_fn_NCO_formal(
    model,
    accelerator,
    inds,
    dataset,
    collate_fn,
    global_batch_size,
    is_main_process,
    N,
    threshold=0.1,
):

    world_size = accelerator.num_processes

    assert global_batch_size % world_size == 0

    # if is_main_process:
    inds = torch.tensor(inds, dtype=int).reshape(world_size, -1).to(accelerator.device)
    # else:
    #     inds = torch.zeros(world_size, global_batch_size // world_size, dtype=int).to(
    #         accelerator.device
    #     )

    factor_val_gather = torch.zeros_like(inds, dtype=float).to(accelerator.device)

    inds_copy = inds.clone()

    # if is_main_process:
    #     import pdb

    #     pdb.set_trace()

    factor_local_copy = torch.zeros_like(
        inds[accelerator.local_process_index], dtype=float
    ).to(accelerator.device)

    model.eval()
    for ii, data_index in enumerate(inds[accelerator.local_process_index]):
        # construct proof traces
        # if accelerator.is_main_process:
        #     import pdb

        #     pdb.set_trace()
        # accelerator.wait_for_everyone()
        proof_trace = dataset[data_index.item()]["traces"]
        # proof trace is list of (state, tactic)
        # construct examples
        tempt_list = []
        for trace in proof_trace:
            tempt_list.append({"state": trace[0], "tactic": trace[1]})
        # collate examples
        cum_log_prob = 0
        micro_batch_size = 4
        for micro_batch_index in range(0, len(tempt_list), micro_batch_size):
            tempt_list_batch = tempt_list[
                micro_batch_index : micro_batch_index + micro_batch_size
            ]
            batch = collate_fn(tempt_list_batch)
            outputs = model(
                input_ids=batch["input_ids"].to(accelerator.device),
                labels=batch["labels"].to(accelerator.device),
                attention_mask=batch["attention_mask"].to(accelerator.device),
            )
            cum_log_prob += log_prob(
                outputs.logits, batch["labels"].to(accelerator.device)
            ).sum()

        factor_local_copy[ii] = fff(torch.exp(cum_log_prob), N).detach()

    # ToDo: Add logging for factor
    # print(f"factor_val: {factor_val}")
    # Set model to train prior to continuing training

    factor_val_gather[accelerator.local_process_index] = factor_local_copy

    factor_val_gather = accelerator.reduce(factor_val_gather, reduction="sum")

    model.train()

    inds_copy = inds_copy.reshape(-1)[factor_val_gather.reshape(-1) > threshold]
    factor_val_gather = factor_val_gather.reshape(-1)[
        factor_val_gather.reshape(-1) > threshold
    ]

    return (
        inds_copy.cpu().tolist(),
        factor_val_gather.float().tolist(),
    )
