"""
AIME 2024 & 2025 benchmark for Qwen3-8B.

Pipeline:
  1. Load model + tokenizer via ModelScope
  2. ========== INSERT MODIFICATION TO MODEL HERE ==========
  3. Wrap with lm_eval's HFLM
  4. Run simple_evaluate on aime24 and/or aime25
  5. Print results

Usage:
  python benchmark_aime.py [--tasks aime24 aime25] [--samples N] [--num-samples K]

Arguments:
  --tasks       Which AIME tasks to run. Choices: aime, aime24, aime25. Default: aime24 aime25
  --limit       Fraction or integer: how many problems to evaluate (useful for quick checks).
                e.g. --limit 5 runs only the first 5 problems. Omit for all 30.
  --num-samples How many generation samples per problem (paper uses 8; default: 1 for quick runs).
  --max-tokens  Max new tokens per generation. Paper uses 32768. Default: 32768.
  --temperature Sampling temperature. Paper uses 0.6. Default: 0.6.
  --top-p       Top-p nucleus sampling. Paper uses 0.95. Default: 0.95.
  --greedy      Use greedy decoding (overrides temperature/top-p, forces num-samples=1).
"""

import argparse
import json
import sys
from pathlib import Path

MODEL_PATH = Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen3-8B"


def parse_args():
    parser = argparse.ArgumentParser(description="AIME benchmark for Qwen3-8B")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["aime24", "aime25"],
        choices=["aime", "aime24", "aime25"],
        help="AIME task(s) to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit number (int) or fraction (float 0-1) of problems to evaluate.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of generation samples per problem (paper: 8, default: 1).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Max new tokens per generation (paper: 32768).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (paper: 0.6).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling (paper: 0.95).",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (sets temperature=0, top_p=1, num_samples=1).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each prompt (after chat template) and raw model response.",
    )
    return parser.parse_args()


def load_model(model_path: Path):
    """Load Qwen3-8B via ModelScope."""
    print(f"Loading model from {model_path} ...")
    from modelscope import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    print(f"  dtype : {model.dtype}")
    print(f"  device: {next(model.parameters()).device}")
    return model, tokenizer


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    #  1. Load model + tokenizer
    # ------------------------------------------------------------------ #
    model, tokenizer = load_model(MODEL_PATH)

    # ------------------------------------------------------------------ #
    #  ================================================================== #
    #  ========== INSERT MODIFICATION TO MODEL HERE ====================  #
    #  ================================================================== #
    #                                                                      #
    #  Example (monkey-patch attention forward for KV eviction):          #
    #                                                                      #
    #    from my_kv_eviction import patch_model                           #
    #    patch_model(model, budget=2048)                                   #
    #                                                                      #
    #  ================================================================== #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  2. Wrap with lm_eval's HFLM
    # ------------------------------------------------------------------ #
    print("Wrapping model with HFLM ...")
    from lm_eval.models.huggingface import HFLM
    import torch

    class VerboseHFLM(HFLM):
        """HFLM subclass that prints the prompt and raw response for every
        generation call when --verbose is set.

        _model_generate() is the single chokepoint in HFLM where tokenised
        context goes in and the full generated token sequence comes out, so
        it's the cleanest place to hook without touching lm_eval internals.
        """

        _call_index = 0  # counts generation calls across the run

        def _model_generate(self, context, max_length, stop, **generation_kwargs):
            VerboseHFLM._call_index += 1
            idx = VerboseHFLM._call_index

            # Decode the prompt (context is a batch of token-id tensors)
            prompt_text = self.tokenizer.decode(
                context[0], skip_special_tokens=False
            )

            sep = "─" * 72
            print(f"\n{sep}")
            print(f"  GENERATION #{idx}  —  prompt tokens: {context.shape[1]}")
            print(sep)
            print(prompt_text)
            print(f"{sep}  [end of prompt]")

            # Run the actual generation
            output = super()._model_generate(
                context, max_length, stop, **generation_kwargs
            )

            # The output includes the prompt tokens; slice them off
            response_ids = output[0][context.shape[1]:]
            response_text = self.tokenizer.decode(
                response_ids, skip_special_tokens=False
            )

            print(f"\n{sep}")
            print(f"  RESPONSE #{idx}  —  new tokens: {len(response_ids)}")
            print(sep)
            print(response_text)
            print(f"{sep}  [end of response]\n")

            return output

    lm = (VerboseHFLM if args.verbose else HFLM)(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,               # AIME needs long context; keep batch=1
        add_bos_token=False,
    )

    # ------------------------------------------------------------------ #
    #  3. Build gen_kwargs
    # ------------------------------------------------------------------ #
    if args.greedy:
        temperature = 0.0
        top_p = 1.0
        num_samples = 1
        print("Decoding: greedy")
    else:
        temperature = args.temperature
        top_p = args.top_p
        num_samples = args.num_samples
        print(
            f"Decoding: temperature={temperature}, top_p={top_p}, "
            f"num_samples={num_samples}"
        )

    gen_kwargs = {
        "max_new_tokens": args.max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0.0,
    }

    # lm_eval repeats each problem `num_samples` times via `num_fewshot`-
    # independent requests; the easiest way to get N samples per problem is
    # to pass `repeats` (supported in newer lm_eval versions) or simply
    # run with `num_samples` inside gen_kwargs.  lm_eval 0.4.x exposes this
    # through a `samples` dict keyed by task name.
    samples_cfg = (
        {task: {"repeats": num_samples} for task in args.tasks}
        if num_samples > 1
        else None
    )

    # ------------------------------------------------------------------ #
    #  4. Run evaluation
    # ------------------------------------------------------------------ #
    import lm_eval

    print(f"\nRunning tasks: {args.tasks}")
    if args.limit is not None:
        print(f"  (limited to {args.limit} problems per task)")

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=0,
        apply_chat_template=True,   # uses Qwen3's instruct chat template
        limit=args.limit,
        gen_kwargs=gen_kwargs,
        samples=samples_cfg,
        log_samples=True,
        verbosity="INFO",
    )

    # ------------------------------------------------------------------ #
    #  5. Print results
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for task_name, task_results in results["results"].items():
        print(f"\n  {task_name}")
        for metric, value in task_results.items():
            if metric.endswith(",none"):
                clean = metric.replace(",none", "")
                print(f"    {clean:<30} {value}")

    print("\n" + "=" * 60)
    print("Paper baselines (Full Attention, Qwen3-8B, 8 samples):")
    print("  aime24   57.1%")
    print("  aime25   40.8%")
    print("=" * 60)

    # Optionally dump full results to JSON
    out_path = Path("benchmark_results.json")
    with open(out_path, "w") as f:
        # results may contain non-serialisable objects; use str fallback
        json.dump(results["results"], f, indent=2, default=str)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
