import torch
import torch.nn.functional as F


def f(x, N):
    x = x.double()
    if N == 1:
        return torch.ones_like(x)
    return torch.where(
        x * N < 0.1,
        (1 - (N - 1) * x + (N - 1) * (N - 2) * x * x / 2)
        / (1 - (N - 1) / 2 * x + (N - 1) * (N - 2) / 6 * x * x),
        N * (1 - x) ** (N - 1) * x / ((1 - (1 - x) ** N)),
    )


def factor(logits, labels, N):

    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    batch_size, seq_len, vocab_size = shift_logits.shape

    # Get the probabilities of the target tokens
    # shift_labels = shift_labels.unsqueeze(-1)  # Shape alignment for gather
    # Replace -100 with 0 temporarily to prevent indexing issues

    # masked_labels = torch.where(
    #     shift_labels == -100, torch.zeros_like(shift_labels), shift_labels
    # )

    # # Gather probabilities and mask out unwanted values
    # token_probs = torch.gather(
    #     torch.log_softmax(shift_logits, dim=-1), -1, masked_labels
    # ).squeeze(
    #     -1
    # )  # Shape: (batch_size, seq_len)

    # # Mask positions where the original labels were -100 to 1.0 (neutral for product)
    # token_probs = torch.where(
    #     shift_labels.squeeze(-1) == -100, torch.zeros_like(token_probs), token_probs
    # )

    token_probs = -F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(batch_size, seq_len)

    # Compute the product for each sequence
    seq_product = torch.sum(token_probs, dim=-1)
    seq_product = torch.exp(seq_product).detach()

    factor = f(seq_product, N).detach()

    return factor

def log_prob(logits, labels):

    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    batch_size, seq_len, vocab_size = shift_logits.shape

    # Get the probabilities of the target tokens
    # shift_labels = shift_labels.unsqueeze(-1)  # Shape alignment for gather
    # Replace -100 with 0 temporarily to prevent indexing issues

    # masked_labels = torch.where(
    #     shift_labels == -100, torch.zeros_like(shift_labels), shift_labels
    # )

    # # Gather probabilities and mask out unwanted values
    # token_probs = torch.gather(
    #     torch.log_softmax(shift_logits, dim=-1), -1, masked_labels
    # ).squeeze(
    #     -1
    # )  # Shape: (batch_size, seq_len)

    # # Mask positions where the original labels were -100 to 1.0 (neutral for product)
    # token_probs = torch.where(
    #     shift_labels.squeeze(-1) == -100, torch.zeros_like(token_probs), token_probs
    # )

    token_probs = -F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(batch_size, seq_len)

    # Compute the product for each sequence
    seq_product = torch.sum(token_probs, dim=-1)

    return seq_product


def custom_loss_function1(logits, labels, N):
    # Compute token probabilities
    shift_logits = logits[..., :-1, :]  # .contiguous()
    shift_labels = labels[..., 1:]  # .contiguous()
    # probs = torch.log_softmax(shift_logits, dim=-1)

    batch_size, seq_len, vocab_size = shift_logits.shape
    # Get the probabilities of the target tokens
    # shift_labels = shift_labels.unsqueeze(-1)  # Shape alignment for gather
    # # Replace -100 with 0 temporarily to prevent indexing issues
    # masked_labels = torch.where(
    #     shift_labels == -100, torch.zeros_like(shift_labels), shift_labels
    # )

    # # Gather probabilities and mask out unwanted values
    # token_probs = torch.gather(probs, -1, masked_labels).squeeze(
    #     -1
    # )  # Shape: (batch_size, seq_len)

    # # Mask positions where the original labels were -100 to 1.0 (neutral for product)
    # token_probs = torch.where(
    #     shift_labels.squeeze(-1) == -100, torch.zeros_like(token_probs), token_probs
    # )

    token_probs = -F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(batch_size, seq_len)

    # Compute the product for each sequence
    seq_product = torch.sum(token_probs, dim=-1)
    seq_product = torch.exp(seq_product).detach()

    factor = f(seq_product, N).detach()

    # Calculate the negative log-likelihood (cross-entropy loss)
    negative_log_likelihood = (
        -token_probs
    )  # The masked token_probs are already zeroed out

    # Normalize by the number of valid tokens (to get mean loss)
    valid_token_mask = (labels[..., 1:] != -100).float()  # Mask for valid positions
    loss = negative_log_likelihood.sum(axis=-1) / valid_token_mask.sum(axis=-1).clamp(
        min=1
    )
    loss = loss * factor
    return loss
