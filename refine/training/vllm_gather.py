import os
import socket
from typing import Tuple

import ray
import ray.actor
import torch
import torch.distributed as dist

# factor libraries better than this
from refine.training.ray_test import create_vllm_engines
from refine.training.ray_dist_utils import init_process_group


def broadcast_weights_from_single_controller(model, engine, model_update_group, modifier_rank, rank):
    """
    This needs to be called by *all* ranks

    A controller process is specified consistent with assignments in `create_vllm_engines`

    NOTE: maybe need an unwrap call here, the broadcast *should* break if not. Can do this outside.
    """
    # avoid OOM
    torch.cuda.empty_cache()

    count, num_params = 0, len(list(model.named_parameters()))
    refss = []
    for name, param in model.named_parameters():
        count += 1
        with GatheredParameters([param], modifier_rank=modifier_rank):
            assert param.data is not None  # ?
            if rank == modifier_rank:
                shape = param.shape
                refs = [
                    engine.update_weight.remote(
                        name,
                        dtype=param.dtype,
                        shape=shape,
                        empty_cache=count == num_params,
                    )
                ]
                refss.extend(refs)
                torch.distributed.broadcast(param.data, 0, group=model_update_group)

    # TODO: can't say I understand what these calls do
    ray.get(refss)


def broadcast_weights_zero_stage_3(model, engine, model_update_group, controller_ids, device_rank, sleep_group):
    # Apparently this needs to be done sequentially, since GatheredParameters can only be called with one
    # modifier rank at a time. It should be fine to just set a barrier with the sleep group.
    # NOTE: confusing, but remember that non-controller devices have `engine` and `model_update_group` None
    for controller_id in controller_ids:
        print(f"Controller {controller_id} is gathering weights and pushing to its VLLM inference engine")
        gather_and_push_weights(model, engine, model_update_group, controller_id, device_rank)
        dist.barrier(group=sleep_group)
        print(f"Controller {controller_id} has completed")


def construct_inference_engine(
    model_name_or_path: str,
    seed: int,
    tensor_parallel_size: int = 4,
    backend: str = "nccl",
    max_model_length: int = 4_096,
    gpu_memory_utilization: float = 0.3,
    group_index: int = 0,
) -> Tuple[dist.distributed_c10d.ProcessGroup, ray.actor.ActorHandle]:
    """
    `backend = "nccl"` requires recent version of vllm

    `tensor_parallel_size` does *not* include the process that broadcasts the model to the engine

    TODO: support instantiating two engines (should be doable with one call)
    """

    # World size for process group containing the main process + the vllm engine workers
    world_size = tensor_parallel_size + 1

    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"

    vllm_engines = create_vllm_engines(
        1,  # n_vllm_engines,
        tensor_parallel_size,
        model_name_or_path,
        seed=seed,
        enable_prefix_caching=False,
        enforce_eager=False,
        max_model_len=max_model_length,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # NOTE: The main process is the one that will be updating the model, in a process group
    # containing this rank and the vllm engine ranks; this process group is separate from
    # the process group doing model training.
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]

    print(
        f"master_address={master_address}, master_port={master_port}, "
        f"world_size={world_size}, backend={backend}"
    )

    group_name = f"openrlhf_{group_index}"

    refs = [
        engine.init_process_group.remote(
            master_address,
            master_port,
            # +1 is important, since distributor is rank 0
            i * tensor_parallel_size + 1,
            world_size,
            group_name,
            backend=backend,
        )
        for i, engine in enumerate(vllm_engines)  # just length 1
    ]

    model_update_group = init_process_group(
        backend=backend,
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)

    return model_update_group, vllm_engines[0]
