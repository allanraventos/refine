import socket
from transformers import (
    AutoModelForCausalLM,
)
import ray
import torch
import torch.distributed

from refine.training.ray_dist_utils import init_process_group
from refine.training.ray_dist_utils import init_logger
from refine.training.ray_test import (
    ray_noset_visible_devices,
    get_all_env_variables,
    LLMRayActor,
    placement_group,
    PlacementGroupSchedulingStrategy,
)


logger = init_logger(__name__)


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    gpu_memory_utilization: float = 0.95,
):
    vllm_engines = []
    # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES will always be set in current context,
    # So we need to get env variables from ray process to check if it is set.
    noset_visible_devices = ray_noset_visible_devices(
        ray.get(get_all_env_variables.remote())
    )
    for i in range(num_engines):
        # When tensor_parallel_size=1 and RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is not set
        # (vLLM mp backend will work smoothly only when *_VISIBLE_DEVICES is modified),
        # vLLM init model in LLMEngine directly, assign 1 GPU for it.
        num_gpus = int(tensor_parallel_size == 1 and not noset_visible_devices)
        scheduling_strategy = None

        if tensor_parallel_size > 1 or noset_visible_devices:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            )

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                name="vllm_engine_0",
            ).remote(
                pretrain,
                noset_visible_devices=noset_visible_devices,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                dtype="bfloat16",
                seed=seed + i,
                enable_prefix_caching=enable_prefix_caching,
                enforce_eager=enforce_eager,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        )

    return vllm_engines


if __name__ == "__main__":

    # torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)

    with open("/home/ubuntu/logs", "a") as log_file:
        print(f"testtestest\n", file=log_file, flush=True)

    with open("/home/ubuntu/logs", "a") as log_file:
        print(f"testtestest\n", file=log_file, flush=True)

    ray.init(log_to_driver=True, namespace="vllm")
    vllm_tensor_parallel_size = 4
    vllm_num_engines = 1
    vllm_sync_backend = "nccl"  # TODO: need later vllm version for nccl

    # model_name_or_path = "allenai/Llama-3.1-Tulu-3-8B-DPO"
    # model_name_or_path2 = "allenai/Llama-3.1-Tulu-3-8B"
    model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_name_or_path2 = "meta-llama/Meta-Llama-3-8B"

    # llm = LLMRayActor.remote("meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=2)
    # output = ray.get(llm.generate.remote("San Franciso is a"))
    # print(f"output: {output}")

    vllm_engines = create_vllm_engines(
        vllm_num_engines,
        vllm_tensor_parallel_size,
        model_name_or_path,
        seed=1,
        enable_prefix_caching=False,
        enforce_eager=False,
        max_model_len=4_096,
        gpu_memory_utilization=0.5,
    )

    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    vllm_num_engines, vllm_tensor_parallel_size = (
        vllm_num_engines,
        vllm_tensor_parallel_size,
    )
    # FIXME: I guess +1 is since there's a machine updating a model and all vllm processes are just doing search with it
    world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
    backend = vllm_sync_backend

    print(
        f"master_address={master_address}, master_port={master_port}, world_size={world_size}, backend={backend}"
    )

    refs = [engine.passs.remote() for i, engine in enumerate(vllm_engines)]

    ray.get(refs)
    # https://github.com/OpenRLHF/OpenRLHF/issues/313
    # if vllm.__version__ > "0.4.2" and os.getenv("NCCL_P2P_DISABLE", "0") == "0":
    #     backend = "gloo"
    #     print(
    #         "Warning: using --vllm_sync_backend=gloo for vLLM version > 0.4.2 (or export NCCL_P2P_DISABLE=1)"
    #     )
    input("Press Enter to terminate the engine...")

    print("ray.get done")
