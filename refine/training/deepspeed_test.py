import os
import time
import torch
import torch.distributed as dist
from torch.distributed import rendezvous


def main():
    # DeepSpeed automatically sets these environment variables
    # e.g. RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", "29501")
    os.environ[""]

    # Initialize the default process group.
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )

    rendezvous_iterator = rendezvous("tcp://127.0.0.1:29502", rank, 5)
    store, rank, world_size = next(rendezvous_iterator)

    if rank == 0:
        print(f"Rank {rank}: Initialized process group on main process.")
        # Sleep to mimic some main-process-only work
        time.sleep(60)

    # Block until rank 0 finishes its work
    dist.barrier()
    print(f"Rank {rank}: Reached the barrier.")


if __name__ == "__main__":
    main()
