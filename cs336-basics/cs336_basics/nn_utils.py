import torch


def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)

def log_softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))


def cross_entropy(inputs, targets):
    negative_log_softmax_logits = -log_softmax(inputs)
    '''
    The torch.gather function is like a selective "picker." It creates a new tensor by picking elements from an input tensor along a specified dimension, using indices provided by another tensor.
    The syntax is: torch.gather(input, dim, index)
        input: The source tensor you want to pick values from. In this case, negative_log_softmax_logits.
        dim: The dimension along which you want to gather values.
        index: A tensor containing the indices to pick. Crucially, the index tensor must have the same number of dimensions as the input tensor.

    Step-by-Step Example
    Let's trace the operation with a simple batch.

    1. Initial Tensors:
    negative_log_softmax_logits (input, shape [4, 5]):
        [[2.3, 1.8, 3.1, 0.9, 1.1],   # Batch item 0
        [0.5, 4.1, 1.2, 2.5, 3.3],   # Batch item 1
        [1.9, 2.2, 0.8, 3.7, 1.4],   # Batch item 2
        [3.9, 1.0, 2.8, 1.5, 0.7]]   # Batch item 3
    targets (indices to pick, shape [4]):
        [2, 4, 1, 0]

    2. Reshape the targets tensor:
    targets.unsqueeze(-1) creates the index tensor.
    Shape changes from [4] to [4, 1].
        [[2],
        [4],
        [1],
        [0]]

    3. Perform the gather operation:
    torch.gather now iterates through the index tensor (targets.unsqueeze(-1)) and picks the corresponding values from the input tensor (negative_log_softmax_logits) along dim=-1.

    For row 0: The index is 2. It picks the value at input[0, 2], which is 3.1.
    For row 1: The index is 4. It picks the value at input[1, 4], which is 3.3.
    For row 2: The index is 1. It picks the value at input[2, 1], which is 2.2.
    For row 3: The index is 0. It picks the value at input[3, 0], which is 3.9.

    4. Final Output:

    The result is a tensor of shape [4, 1] containing the selected loss values for each item in the batch.

        [[3.1],
        [3.3],
        [2.2],
        [3.9]]
    '''
    return torch.mean(torch.gather(negative_log_softmax_logits, -1, targets.unsqueeze(-1)))


def clip_gradient(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = 0.0

    for g in grads:
        norm += (g**2).sum()

    norm = torch.sqrt(norm)
    clip_coef = min(1, max_norm / (norm + 1e-6))
    for g in grads:
        g *= clip_coef
