"""One-clip inference from a trained Task 1 checkpoint (spec S10.5).

The demo notebook needs an end-to-end example: raw text in, predicted tags out,
using the weights that produced the reported numbers rather than a fresh model.

    python src/infer.py --checkpoint artifacts/checkpoints/task1_musiccaps.pt \
        --text "A gentle piano ballad with soft female vocals."

Runs on CPU in a couple of seconds. The checkpoint carries its own tag
vocabulary and the decision threshold selected on validation, so nothing about
the setup is re-specified here and the notebook cannot silently disagree with
the reported run.
"""

import argparse

import torch

from src import bert_encoder


def load_tagger(checkpoint_path: str, device: str = "cpu"):
    """Returns (model, tokenizer, vocab, threshold) ready for `predict`."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    cfg = ckpt.get("config", {})
    model = bert_encoder.BertTagger(
        n_tags=len(vocab),
        model_name=cfg.get("model", bert_encoder.DEFAULT_MODEL),
        pooling=cfg.get("pooling", "cls"),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    tokenizer = bert_encoder.AutoTokenizer.from_pretrained(
        cfg.get("model", bert_encoder.DEFAULT_MODEL)
    )
    return model, tokenizer, vocab, ckpt.get("threshold", 0.5)


def predict(model, tokenizer, vocab, text: str, max_length: int = 128,
            device: str = "cpu") -> list[tuple[str, float]]:
    """Tag probabilities for one piece of text, highest first."""
    batch = tokenizer(
        text, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(
            model(batch["input_ids"], batch["attention_mask"])
        )[0].tolist()
    return sorted(zip(vocab, probs), key=lambda kv: -kv[1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/checkpoints/task1_musiccaps.pt")
    p.add_argument("--text", required=True)
    p.add_argument("--top", type=int, default=8)
    args = p.parse_args()

    model, tokenizer, vocab, thr = load_tagger(args.checkpoint)
    ranked = predict(model, tokenizer, vocab, args.text)
    print("input: %s\n" % args.text)
    print("%-22s %7s %s" % ("tag", "p", "predicted at thr=%.2f" % thr))
    for tag, p_ in ranked[: args.top]:
        print("%-22s %7.3f %s" % (tag, p_, "<--" if p_ >= thr else ""))


if __name__ == "__main__":
    main()
