"""Fine-tune the chat model on my replies on a Colab GPU, then answer the held-out moments.

Runs inside a Colab kernel after `setup.sh`, with the files `shinnku corpus sft`
writes in SFT_DATA. It trains a LoRA with Unsloth on train.jsonl (loss on my reply
only), watches the loss on valid.jsonl, saves the adapter, and writes one
{"id", "reply"} line per moment of eval.jsonl / eval_recall.jsonl for
`shinnku eval run --replies`. It prints sizes, losses and timings, never text.

Settings are SFT_* environment variables (`colab exec --env`); see CONFIG.
plans/finetune-plan.md has the colab commands.
"""

import json
import os
import time
from pathlib import Path

CONFIG = {
    "SFT_MODEL": "unsloth/Qwen3.6-35B-A3B",
    "SFT_DATA": "/content/sft",
    "SFT_OUT": "/content/out",
    "SFT_EPOCHS": "2",
    "SFT_RANK": "32",
    "SFT_LR": "1e-4",
    "SFT_MAX_LEN": "4096",
    "SFT_BATCH": "2",
    "SFT_ACCUM": "8",
    # A dry run on a small model: stop after this many steps, train on this many examples.
    "SFT_MAX_STEPS": "-1",
    "SFT_LIMIT": "0",
}
END = "<|im_end|>"


def setting(name: str) -> str:
    return os.environ.get(name, CONFIG[name])


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_text(tokenizer, messages: list[dict]) -> str:
    """The prompt as the model sees it when answering, thinking switched off."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def encode(tokenizer, row: dict) -> dict:
    """Token ids of prompt and reply; only the reply (and its end mark) is learned."""
    prompt = tokenizer(prompt_text(tokenizer, row["messages"]), add_special_tokens=False)["input_ids"]
    reply = tokenizer(row["reply"] + END, add_special_tokens=False)["input_ids"]
    return {"input_ids": prompt + reply, "labels": [-100] * len(prompt) + reply}


def describe(name: str, lengths: list[int]) -> None:
    if not lengths:
        print(f"{name}: none")
        return
    ordered = sorted(lengths)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    print(
        f"{name}: n={len(lengths)} tokens mean={sum(lengths) // len(lengths)} "
        f"p95={p95} max={ordered[-1]} total={sum(lengths)}"
    )


def train(model, tokenizer, data: Path, out: Path):
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from unsloth import is_bf16_supported

    max_len = int(setting("SFT_MAX_LEN"))
    limit = int(setting("SFT_LIMIT"))

    def dataset(name: str) -> Dataset:
        rows = read(data / name)
        rows = rows[:limit] if limit else rows
        encoded = [encode(tokenizer, row) for row in rows]
        kept = [row for row in encoded if len(row["input_ids"]) <= max_len]
        describe(f"{name} (dropped {len(encoded) - len(kept)} over {max_len})", [len(r["input_ids"]) for r in kept])
        return Dataset.from_list(kept)

    train_set, valid_set = dataset("train.jsonl"), dataset("valid.jsonl")
    tokenizer.padding_side = "right"
    every = 100
    checkpoints = out / "checkpoints"
    args = TrainingArguments(
        output_dir=str(checkpoints),
        per_device_train_batch_size=int(setting("SFT_BATCH")),
        per_device_eval_batch_size=int(setting("SFT_BATCH")),
        gradient_accumulation_steps=int(setting("SFT_ACCUM")),
        num_train_epochs=float(setting("SFT_EPOCHS")),
        max_steps=int(setting("SFT_MAX_STEPS")),
        learning_rate=float(setting("SFT_LR")),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        optim="adamw_8bit",
        bf16=is_bf16_supported(),
        fp16=not is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=every,
        save_steps=every,
        save_total_limit=2,
        disable_tqdm=True,
        report_to="none",
        seed=3407,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    started = time.monotonic()
    # Picks up where an earlier run on this VM stopped.
    trainer.train(resume_from_checkpoint=any(checkpoints.glob("checkpoint-*")))
    took = time.monotonic() - started
    tokens = sum(len(ids) for ids in train_set["input_ids"]) * float(setting("SFT_EPOCHS"))
    print(f"trained in {took / 60:.1f} min, about {tokens / took:.0f} tokens/s")
    print(f"valid loss: {trainer.evaluate()['eval_loss']:.4f}")
    model.save_pretrained(str(out / "lora"))
    tokenizer.save_pretrained(str(out / "lora"))


def answer(model, tokenizer, prompts: Path, path: Path, batch: int = 8) -> None:
    """One sampled reply per moment, as the bot would be asked for it."""
    import torch

    rows = sorted(read(prompts), key=lambda row: len(prompt_text(tokenizer, row["messages"])))
    tokenizer.padding_side = "left"
    end = tokenizer.convert_tokens_to_ids(END)
    started = time.monotonic()
    with path.open("w", encoding="utf-8") as out:
        for start in range(0, len(rows), batch):
            chunk = rows[start : start + batch]
            inputs = tokenizer(
                [prompt_text(tokenizer, row["messages"]) for row in chunk],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to("cuda")
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    eos_token_id=end,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new = output[:, inputs["input_ids"].shape[1] :]
            for row, ids in zip(chunk, new):
                reply = tokenizer.decode(ids, skip_special_tokens=True).strip()
                out.write(json.dumps({"id": row["id"], "reply": reply}, ensure_ascii=False) + "\n")
    print(f"{path.name}: {len(rows)} replies in {time.monotonic() - started:.0f}s")


def main() -> None:
    from unsloth import FastLanguageModel

    data, out = Path(setting("SFT_DATA")), Path(setting("SFT_OUT"))
    out.mkdir(parents=True, exist_ok=True)
    rank = int(setting("SFT_RANK"))
    model, processor = FastLanguageModel.from_pretrained(
        setting("SFT_MODEL"),
        max_seq_length=int(setting("SFT_MAX_LEN")),
        load_in_4bit=False,  # Unsloth advises against 4-bit for these models
        fast_inference=False,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=2 * rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "gate_up_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        bias="none",
    )
    train(model, tokenizer, data, out)
    FastLanguageModel.for_inference(model)
    for name in ("eval", "eval_recall"):
        if (data / f"{name}.jsonl").is_file():
            answer(model, tokenizer, data / f"{name}.jsonl", out / f"{name}_out.jsonl")


if __name__ == "__main__":
    main()
