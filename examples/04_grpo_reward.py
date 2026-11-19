"""Reward functions for RL fine-tuning (GRPO, PPO, RLOO, best-of-n).

swadesh.rewards returns plain Python callables. Each one takes the batch of
completions plus the reference column as keyword arguments and returns one
float per completion, scaled to [0, 1] by default. This file calls them
directly, so it runs without TRL installed.

TRL wiring (not run here, because TRL is not a dependency of swadesh):

    from trl import GRPOConfig, GRPOTrainer
    from swadesh.rewards import chrf_reward, bleu_reward

    trainer = GRPOTrainer(
        model="your/model",
        reward_funcs=[chrf_reward(ref_column="reference")],
        args=GRPOConfig(output_dir="out"),
        train_dataset=dataset,   # must have a "reference" column
    )

TRL passes the dataset columns to reward functions as keyword arguments, so
`ref_column` has to match a column name. Check the argument names against the
TRL version you use.

Run: python examples/04_grpo_reward.py
"""
from swadesh.rewards import bleu_reward, chrf_reward

reward = chrf_reward(ref_column="reference")

# Plain completions: one string per sample.
completions = [
    "Das ist sehr gut.",
    "Das ist gut.",
    "Katze auf der Matte.",
]
references = ["Das ist sehr gut.", "Das ist sehr gut.", "Die Katze sitzt auf der Matte."]
print("chrF reward:", [round(r, 3) for r in reward(completions=completions, reference=references)])

# Conversational completions: a list of messages per sample, as TRL passes
# them for chat datasets. The last message's content is scored.
chat_completions = [
    [{"role": "assistant", "content": "Das ist sehr gut."}],
    [{"role": "assistant", "content": "Das ist gut."}],
]
print("chrF reward (chat format):",
      [round(r, 3) for r in reward(completions=chat_completions, reference=references[:2])])

# chrF++ and sentence BLEU are available the same way.
print("chrF++ reward:", [round(r, 3) for r in chrf_reward(word_order=2)(
    completions=completions, reference=references)])
print("BLEU reward:", [round(r, 3) for r in bleu_reward()(
    completions=completions, reference=references)])

# Several references per example: pass a list of references for each sample,
# and each reference stream becomes one entry in the corpus layout.
multi_ref = [["Das ist sehr gut.", "Das ist wirklich gut."]] * 3
print("chrF reward, two refs:", [round(r, 3) for r in reward(completions=completions, reference=multi_ref)])
