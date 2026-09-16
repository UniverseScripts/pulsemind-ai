"""Local text generation for the explanation layer -- the model loader.

Mirrors scoring.py. One place owns HOW the model is loaded, because model
identity includes the quantisation and the device, not just the checkpoint
name: the same weights in NF4 and in fp16 are two different generators, and an
explanation must carry which one produced it.

LOCAL ONLY
----------
The golden set is MIMIC-derived and DUA-covered, so handing it to a hosted API
would redistribute credentialed data. Nothing in this module can reach a
network except the one-time weight download from the Hub.

GREEDY, SEEDED, NOT SAMPLED
---------------------------
do_sample=False and a fixed seed. This is not a quality choice -- it is what
makes a prompt change attributable. The whole point of grounding.py is to say
"this edit removed that violation", and a generator that answers differently on
the same input each run cannot support that claim.

WHAT THE GENERATOR IS
---------------------
Exactly the `generator(payload) -> str` callable that explain.explain() already
expects. Nothing here decides WHETHER a record may be explained; the
sufficiency gate lives upstream in explain.build_payload(), which raises rather
than returning something a generator could consume.

⚠️ transformers MUST BE < 5
---------------------------
transformers 5.14.1 SEGFAULTS (exit 139) loading a 7B in NF4 on this card. Not
an exception, a hard crash in native code, and it reproduces across device_map
"auto" and {"": 0}, with and without double quantisation, on both float16 and
bfloat16 compute. The same stack loads a 0.5B fine, so it presents as a size
problem and is easy to misattribute to bitsandbytes.

It is not bitsandbytes. On transformers 4.57.6 the identical 7B NF4 load
succeeds in 23 s at 5.57 GB VRAM, same bitsandbytes 0.50.0, same driver, same
weights. Pin below 5 until that is fixed upstream.

Two other Windows-specific notes from the same investigation:
  * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is a NO-OP here -- torch
    warns "not supported on this platform". It cannot be used to work around
    allocator fragmentation on Windows.
  * A failed load can surface as OSError 1455 "the paging file is too small"
    rather than as an OOM. That is commit exhaustion, not a GPU problem; check
    what else is holding RAM before touching the model configuration.
"""
from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field

from .. import config as C
from . import explain as E


def _quant_config(quant: str):
    """NF4 config, or None for fp16. `none` is the documented fallback for a
    smaller model if 4-bit ever stops working on this card."""
    if quant == "none":
        return None
    if quant != "nf4":
        raise ValueError(f"unknown LLM_QUANT {quant!r}; expected 'nf4' or 'none'")
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)


def _host_embedding(model, compute_device) -> None:
    """Move the embedding table into host RAM and keep it there.

    ⚠️ NOT VIA `device_map`, and the reason matters.

    `{"model.embed_tokens": "cpu", "": 0}` resolves correctly -- accelerate walks
    a parameter name from the longest prefix down (`utils/modeling.py:1991`) -- so
    it looks exactly like the right answer. It is not. In a map whose main device
    is a GPU, `cpu` means OFFLOAD, not "run there":

        offloaded_devices = ["disk"] if main_device == "cpu" ... else ["cpu", "disk"]
        offload = {name: device in offloaded_devices ...}   # big_modeling.py:405

    The module is left on `meta`, its execution device is set to the GPU, and
    `AlignDevicesHook` streams the weight ONTO the card for every forward and
    frees it after. For a 1040 MiB table across a prefill and 220 decode steps
    that is the wrong trade twice over: the copies dominate generation, and the
    table is resident again at exactly the moment the card is fullest. Measured
    here it did not finish loading a 0.5B in seven minutes.

    So: load normally, then move the module and let two hooks handle the crossing.
    Public torch API, no accelerate internals, and the table crosses the bus once
    at startup instead of once per token.

    What travels instead is the ACTIVATIONS -- `seq_len x 3584 x 2` bytes on the
    prefill, 7 KB per decode step. Three orders of magnitude less.

    ⚠️ This lowers what the model HOLDS, not what the load PEAKS at. The table is
    on the card while `from_pretrained` runs, so the VRAM gate -- which protects a
    load -- cannot come down on the strength of this alone.

    ⚠️ AND THE DRIVER BARELY NOTICES, WHICH IS NOT THE SAME AS IT NOT WORKING.
    Measured: torch's `allocated` falls by 1039 MiB and the driver gets 130 back,
    because a freed block inside a partially-used segment needs
    `expandable_segments` to be returned and that is a no-op on Windows. Read
    there and this looks worthless. The other 909 MiB is REUSABLE ARENA:
    generation puts its KV cache and activations inside it instead of asking the
    driver for new segments, so the card ends a generation with room on it.
    Three loads each way -- `embed=cuda` finished with 119/97/117 MiB free in
    25.6/17.8/21.6 s; `embed=cpu` with 770/692/678 MiB free in 14.4/13.2/13.3 s.
    Generation swings ~5x with what is left on the card, and this is what leaves
    something on it.

    ⚠️ AFTER THIS RUNS, `model.device` IS A LIE. `PreTrainedModel.device` reports
    the device of the FIRST parameter, and `embed_tokens` is the first parameter,
    so a model whose compute lives entirely on the GPU starts answering "cpu".
    Anything that sends inputs to `model.device` then sends them to the host: the
    ids land on the CPU, `cache_position` and `position_ids` are derived from
    THEIR device, and the rotary embedding dies on `mat2 is on cpu`. Measured --
    the load succeeded and generation failed on the first forward.

    Hence `compute_device`, threaded explicitly and carried on `Generator`. The
    load knows the answer before the move makes the question ambiguous.
    """
    import torch

    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()

    # ⚠️ TIED WEIGHTS SHARE ONE STORAGE, so moving the input embedding moves
    # `lm_head` with it -- silently, because they are the same Parameter object.
    # `lm_head` runs a 152,064-way matmul every decode step and would then run it
    # on the CPU. This model unties (`tie_word_embeddings: false`), but the
    # documented fallback in config is "a 3B model", and every smaller Qwen2.5
    # ties. That is a reachable path, so it refuses rather than degrades.
    if out is not None and out.weight is emb.weight:
        # Named from the LOADED model, not from config: they agree in production
        # and diverge exactly when someone is testing an override, which is the
        # one moment a wrong model name in the message costs real time.
        raise RuntimeError(
            f"{getattr(model, 'name_or_path', None) or C.LLM_MODEL_ID} ties its "
            f"input and output embeddings, so moving the table would put lm_head "
            f"on the CPU as well. Set PM_LLM_EMBED_DEVICE=cuda for this model.")

    emb.to("cpu")
    # The ids are built on the GPU by `generate`; the gather has to happen beside
    # the table, and the activations have to come back.
    emb.register_forward_pre_hook(lambda _m, args: (args[0].to("cpu"),) + args[1:])
    emb.register_forward_hook(
        lambda _m, _args, out: out.to(compute_device, non_blocking=True))
    # Without this the allocator keeps the 1040 MiB and the driver still reports
    # it used -- the saving would be real in torch's books and invisible in the
    # only figure the gate reads.
    torch.cuda.empty_cache()


@dataclass(slots=True)
class Generator:
    model: object
    tokenizer: object
    provenance: dict
    #: Where the compute lives. NOT `model.device` -- with the embedding table on
    #: the host that property reports the first parameter's device, which is the
    #: host, and every input would follow it there. Defaults to None so an older
    #: caller constructing a Generator positionally still works; `__call__` then
    #: falls back to the old behaviour, which is correct whenever nothing moved.
    device: object = None
    max_new_tokens: int = C.LLM_MAX_NEW_TOKENS
    latencies: list = field(default_factory=list)

    def __call__(self, payload: dict) -> str:
        import torch
        msgs = E.render_prompt(payload)
        enc = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(
                self.device if self.device is not None else self.model.device)
        n_in = enc["input_ids"].shape[1]
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        self.latencies.append(time.time() - t0)
        # Only the completion. Decoding the whole sequence would hand the
        # verifier the system prompt as well, and the system prompt names every
        # band -- which would trip BAND_MISMATCH on text that never said it.
        return self.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()


def vram_status() -> dict:
    """Free and total VRAM, from the driver rather than from CUDA's view of it.

    ⚠️ `torch.cuda.mem_get_info()` reports what the CUDA runtime believes a new
    allocation could get, which on this WDDM setup is not what the device
    actually has free. Measured 2026-08-19 with the 7B already resident:

        nvidia-smi   260 MiB free of 8151
        torch       6759 MiB free of 8151

    A 6.3 GB disagreement, and torch is the optimistic one. That is precisely how
    a preflight passes and the load then segfaults -- `s19_generate` gates on
    `vram_free_gb > 5.5`, which torch would have cleared with a quarter of a
    gigabyte actually available. NVML talks to the driver; CUDA talks to its own
    bookkeeping. Prefer the driver, and say which one answered.

    Totals agree exactly and always did: 8151 MiB is 7.96 GiB is 8.55 decimal GB.
    The card is an 8 GB card; every other figure in the notes was that same
    number with its units mangled.
    """
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            raw = subprocess.run(
                [exe, "--query-gpu=memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True).stdout
            total_mib, free_mib = (int(v) for v in raw.splitlines()[0].split(","))
            return {"free_mib": free_mib, "total_mib": total_mib, "source": "nvidia-smi"}
        except Exception:  # noqa: BLE001
            pass  # fall through -- an unreadable driver is not a reason to stop

    import torch
    free, total = torch.cuda.mem_get_info()
    return {"free_mib": round(free / 1024**2), "total_mib": round(total / 1024**2),
            # Named so it shows up in the report: this number is the optimistic
            # one, and a reader should know the driver was not available.
            "source": "torch.cuda.mem_get_info (nvidia-smi unavailable)"}


def capability_check() -> dict:
    """The cheap check the stage runs every time: is the toolchain present and
    is there room?

    Deliberately does NOT load a model. Loading a probe model in the same
    process as the real one leaves a CUDA context and fragmented segments
    behind, and a 7B in NF4 needs a ~4.8 GB block on a card that only has ~6.4
    GB free once the display has taken its share -- the probe was the reason the
    real load then failed on fragmentation. The full probe lives in preflight(),
    which is a separate command.
    """
    import torch
    import huggingface_hub.constants as K

    # WHERE THE WEIGHTS WILL LAND, asserted rather than assumed.
    #
    # config.py sets HF_HOME, and that resolves correctly when tested. It did
    # not hold on 2026-08-08: 14.19 GB of Qwen weights appeared under
    # C:\Users\...\.cache\huggingface\hub, a full second copy beside the 15.23 GB
    # already on D:, and took the system drive to 1.23 GB free. The trigger was
    # never reproduced -- it coincided with huggingface_hub going 1.27.0 -> 0.36.2
    # on a transformers downgrade.
    #
    # An unreproducible cause is exactly the kind worth guarding rather than
    # explaining. Downloading 14 GB to the wrong drive should be a loud failure,
    # not something discovered when the disk fills.
    hub = pathlib.Path(K.HF_HUB_CACHE).resolve()
    if not str(hub).lower().startswith(str(pathlib.Path(C.LLM_CACHE).resolve()).lower()):
        raise RuntimeError(
            f"the model cache resolves to {hub}, not {C.LLM_CACHE}. Weights are "
            f"~15 GB and this would fill the system drive. Set HF_HOME (and "
            f"HF_HUB_CACHE) explicitly before importing transformers.")

    p = torch.cuda.get_device_properties(0)
    # From the driver, not from CUDA's bookkeeping -- see `vram_status`. The
    # `_gb` keys keep their names and decimal-GB units because `s19_generate`
    # gates on `vram_free_gb`; what changed is that they are now true.
    vram = vram_status()
    out = {"gpu": p.name, "compute_capability": f"sm_{p.major}{p.minor}",
           "vram_free_gb": round(vram["free_mib"] * 1024**2 / 1e9, 2),
           "vram_total_gb": round(vram["total_mib"] * 1024**2 / 1e9, 2),
           "vram_free_mib": vram["free_mib"], "vram_total_mib": vram["total_mib"],
           "vram_source": vram["source"],
           "hub_cache": str(hub),
           "quantisation": C.LLM_QUANT, "torch": torch.__version__}
    if C.LLM_QUANT != "none":
        import bitsandbytes
        out["bitsandbytes"] = bitsandbytes.__version__
    return out


def preflight(model_id: str | None = None) -> dict:
    """Prove the toolchain works before committing to a multi-GB download.

    bitsandbytes needs sm_120 kernels and Blackwell support is recent enough to
    be version-sensitive. A ~350 MB probe that quantises and generates one token
    costs seconds; discovering the same failure after pulling 5 GB costs an hour
    and tells you nothing extra.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid = model_id or C.LLM_PREFLIGHT_MODEL
    p = torch.cuda.get_device_properties(0)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(
        mid, quantization_config=_quant_config(C.LLM_QUANT), device_map={"": 0})
    model.eval()
    enc = tok.apply_chat_template([{"role": "user", "content": "Reply with exactly: ok"}],
                                  add_generation_prompt=True, tokenize=True,
                                  return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    vram = torch.cuda.memory_allocated() / 1e9
    del model
    torch.cuda.empty_cache()
    return {"probe_model": mid, "device": f"sm_{p.major}{p.minor}", "gpu": p.name,
            "quantisation": C.LLM_QUANT, "vram_gb": round(vram, 2),
            "seconds": round(time.time() - t0, 1), "output": txt}


def load_generator() -> Generator:
    """The model, the tokenizer, and the provenance that must travel with every
    line it writes."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(C.LLM_SEED)
    tok = AutoTokenizer.from_pretrained(C.LLM_MODEL_ID, revision=C.LLM_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        C.LLM_MODEL_ID, revision=C.LLM_REVISION,
        quantization_config=_quant_config(C.LLM_QUANT), device_map={"": 0})
    model.eval()

    # Read BEFORE anything moves: every parameter is still on the card here, so
    # this is unambiguous. After `_host_embedding` it would not be.
    compute_device = model.device

    if C.LLM_EMBED_DEVICE == "cpu":
        _host_embedding(model, compute_device)
    elif C.LLM_EMBED_DEVICE != "cuda":
        raise ValueError(
            f"unknown LLM_EMBED_DEVICE {C.LLM_EMBED_DEVICE!r}; expected 'cpu' or 'cuda'")

    # WHERE THE EMBEDDING ACTUALLY LANDED, asserted rather than assumed.
    #
    # A move that silently failed produces a model that loads, generates
    # correctly, and quietly holds the 1040 MiB this setting exists to free. A
    # saving that did not happen must not look like one that did.
    embed = model.get_input_embeddings().weight
    want = "cpu" if C.LLM_EMBED_DEVICE == "cpu" else "cuda"
    if embed.device.type != want:
        raise RuntimeError(
            f"LLM_EMBED_DEVICE={C.LLM_EMBED_DEVICE} but embed_tokens is on "
            f"{embed.device}; the move did not take.")
    gpu = torch.cuda.get_device_properties(0)

    prov = {
        "model": C.LLM_MODEL_ID,
        "revision": C.LLM_REVISION,
        "quantisation": C.LLM_QUANT,
        "dtype": str(model.dtype),
        # The COMPUTE device, read before the embedding moved. `model.device`
        # would now name the host and misreport where this model actually runs.
        "device": str(compute_device),
        # Part of model identity for the same reason `quantisation` is: it says
        # which machine configuration wrote this line. It is deliberately NOT in
        # `FP_GENERATE` -- an embedding lookup is a gather, and bf16 round-trips
        # through the host exactly, so moving it is provably output-neutral and
        # fingerprinting it would force a re-run that changes nothing. That
        # argument rests on a hash: `tools/vram_probe.py` reports one per run and
        # the two placements must agree. If they ever stop agreeing, this is no
        # longer a placement detail and belongs in the fingerprint.
        "embed_device": str(model.get_input_embeddings().weight.device),
        "compute_capability": f"sm_{gpu.major}{gpu.minor}",
        "max_new_tokens": C.LLM_MAX_NEW_TOKENS,
        "seed": C.LLM_SEED,
        "decoding": "greedy",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    if C.LLM_QUANT != "none":
        import bitsandbytes
        prov["bitsandbytes"] = bitsandbytes.__version__
    return Generator(model=model, tokenizer=tok, provenance=prov,
                     device=compute_device)
