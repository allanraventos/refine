import torch
from torch import nn


def f(x, N):
    if N == 1:
        return torch.ones_like(x)
    return torch.where(
        x * N < 0.1,
        (1 - (N - 1) * x + (N - 1) * (N - 2) * x * x / 2)
        / (1 - (N - 1) / 2 * x + (N - 1) * (N - 2) / 6 * x * x),
        N * (1 - x) ** (N - 1) * x / ((1 - (1 - x) ** N)),
    )


def custom_loss_function2(logits, labels, N):
    # Compute token probabilities
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    probs = torch.log_softmax(shift_logits, dim=-1)
    # Get the probabilities of the target tokens
    shift_labels = shift_labels.unsqueeze(-1)  # Shape alignment for gather
    # Replace -100 with 0 temporarily to prevent indexing issues
    masked_labels = torch.where(
        shift_labels == -100, torch.zeros_like(shift_labels), shift_labels
    )

    # Gather probabilities and mask out unwanted values
    token_probs = torch.gather(probs, -1, masked_labels).squeeze(
        -1
    )  # Shape: (batch_size, seq_len)

    # Mask positions where the original labels were -100 to 1.0 (neutral for product)
    token_probs = torch.where(
        shift_labels.squeeze(-1) == -100, torch.zeros_like(token_probs), token_probs
    )

    # Compute the product for each sequence
    seq_product = torch.sum(token_probs, dim=-1)
    seq_product = torch.exp(seq_product).detach()

    factor = f(seq_product, N).detach()

    # Calculate the negative log-likelihood (cross-entropy loss)
    negative_log_likelihood = (
        -token_probs
    )  # The masked token_probs are already zeroed out

    # Normalize by the number of valid tokens (to get mean loss)
    # valid_token_mask = (labels[..., 1:] != -100).float()  # Mask for valid positions
    loss = negative_log_likelihood.sum(axis=-1)  # / valid_token_mask.sum(axis=-1)
    loss = loss * factor
    return loss.mean()


def compute_mean_loss_per_example(
    model_outputs,
    labels: torch.Tensor,
    shift_labels: bool = True,
):
    """
    Computes a per-example (scalar) loss for causal LM, ignoring:
      - any positions labeled -100 (prompt tokens, etc.),
      - pad tokens (converted to -100 here).
    Returns: a 1D tensor [batch_size] with the mean loss per example.

    NOTE: this is only needed because we want to be able to keep losses
    only for hard examples (TODO: for which we need to check for exact match)

    Arguments:
      model: A causal LM model (e.g. from AutoModelForCausalLM).
      input_ids: [batch_size, seq_len], possibly padded.
      labels: [batch_size, seq_len], where *some* positions may already be -100 (prompt).
      attention_mask: [batch_size, seq_len], optional.
      shift_labels: Whether to shift last logit & first label for causal LM
                    (typical next-token prediction).
    """

    # 1) Forward pass (disable built-in labels=... to avoid auto-loss):
    logits = model_outputs.logits  # shape: [B, S, vocab_size]

    # 2) Shift for causal LM (drop last logit, drop first label)
    if shift_labels:
        logits = logits[..., :-1, :].contiguous()  # shape: [B, S-1, vocab_size]
        labels = labels[..., 1:].contiguous()  # shape: [B, S-1]

    # 4) Compute token-level losses with reduction="none" and ignore_index=-100
    loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    # Flatten batch & seq for cross entropy
    vocab_size = logits.size(-1)
    token_losses = loss_fn(
        logits.view(-1, vocab_size),  # [B*(S-1), vocab_size]
        labels.view(-1),  # [B*(S-1)]
    )
    # Reshape back to [B, S-1]
    token_losses = token_losses.view(labels.size(0), labels.size(1))

    # 5) Compute mean loss per example, ignoring positions labeled -100
    #    (They are automatically assigned 0 in 'token_losses', but let's be explicit.)
    valid_mask = labels != -100  # True/False for valid tokens
    sum_per_example = (token_losses * valid_mask).sum(dim=1)
    valid_counts = valid_mask.sum(dim=1).clamp(min=1)  # avoid division by 0
    mean_loss_per_example = sum_per_example / valid_counts

    sum_all = (token_losses * valid_mask).sum()
    count_all = valid_mask.sum()
    global_avg_loss = sum_all / count_all

    return mean_loss_per_example, global_avg_loss


def custom_eval_function1(logits, labels, N):
    # Compute token probabilities
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    probs = torch.log_softmax(shift_logits, dim=-1)
    # Get the probabilities of the target tokens
    shift_labels = shift_labels.unsqueeze(-1)  # Shape alignment for gather
    # Replace -100 with 0 temporarily to prevent indexing issues
    masked_labels = torch.where(
        shift_labels == -100, torch.zeros_like(shift_labels), shift_labels
    )

    # Gather probabilities and mask out unwanted values
    token_probs = torch.gather(probs, -1, masked_labels).squeeze(
        -1
    )  # Shape: (batch_size, seq_len)

    # Mask positions where the original labels were -100 to 1.0 (neutral for product)
    token_probs = torch.where(
        shift_labels.squeeze(-1) == -100, torch.zeros_like(token_probs), token_probs
    )

    # Compute the product for each sequence
    seq_product = torch.sum(token_probs, dim=-1)
    seq_product = torch.exp(seq_product).detach()

    # Calculate the negative log-likelihood (cross-entropy loss)
    negative_log_likelihood = (
        -token_probs
    )  # The masked token_probs are already zeroed out

    # Normalize by the number of valid tokens (to get mean loss)
    valid_token_mask = (labels[..., 1:] != -100).float()  # Mask for valid positions
    loss = negative_log_likelihood.sum(axis=-1)
    return loss
