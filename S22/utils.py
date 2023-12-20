
import torch
import random
import torch.nn as nn
import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import FSDPStrategy

from tsai_gpt.model import GPT, Block, Config
from tsai_gpt.tokenizer import Tokenizer
from tsai_gpt.utils import get_default_supported_precision, load_checkpoint, gptq_quantization


example_text = [
    "In a galaxy far, far away, an intergalactic council convenes to discuss the rising cost of lightsaber batteries. Among them is an unlikely representative: a droid with a penchant for economics...",
    "As Sherlock Holmes and Dr. Watson enter the world of social media influencers, they find their first case: the mysterious disappearance of a famous TikTok star's like button.",
    "In the midst of a zombie apocalypse, a group of survivors discovers a library with every book intact except for cookbooks. Their leader, a former TV chef, decides to write the ultimate survival recipe book titled...",
    "A time traveler accidentally attends Shakespeare's first play, but instead of a quill, she hands him a smartphone with autocorrect. The resulting play is...",
    "Amidst the chaos of a Hogwarts School reunion, a magical mishap swaps the voices of Professors Dumbledore and Snape, leading to an unexpected duet in the Great Hall that goes viral in the wizarding world."
]

examples = [
             [
                example_text[i], 
                round(random.uniform(0.7,1), 1), 
                int(random.uniform(120,200)), 
                int(random.uniform(200,300))] for i,x in enumerate(example_text)
           ]


model_name = "pythia-160m"
name = "redpajama"

checkpoint_dir = Path("iter-010915-ckpt.pth")
quantize = None
strategy = "auto"
devices = 1
precision = get_default_supported_precision(training=False)
plugins = None
fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy, plugins=plugins)
fabric.launch()


with fabric.init_module(empty_init=True), gptq_quantization(quantize=="gptq.int4"):
    config = Config.from_name(model_name)
    model = GPT(config)

model.eval()
model = fabric.setup_module(model)
load_checkpoint(fabric, model, checkpoint_dir)

tokenizer = Tokenizer(Path('tokenizer'))


def generate_dialogue(input_text, temperature, max_tokens, top_k):
    encoded = tokenizer.encode(input_text, device=fabric.device)
    max_returned_tokens = encoded.size(0) + max_tokens


    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens


    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)

    y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k)

    return(tokenizer.decode(y))


@torch.inference_mode()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k:int = None,
    eos_id:int = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    T = idx.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)

    # generate up to a fixed number of tokens
    for _ in range(max_returned_tokens - T):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx
