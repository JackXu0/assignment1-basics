"""Training loop for a Transformer language model."""

import argparse
import os
import time

import numpy as np
import torch
import wandb

from cs336_basics.building_blocks import (
    AdamW,
    TransformerLM,
    cross_entropy,
    get_batch,
    gradient_clipping,
    lr_cosine_schedule,
    save_checkpoint,
    load_checkpoint,
)


def evaluate(model, data, batch_size, context_length, device, eval_iters):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_iters):
            inputs, targets = get_batch(data, batch_size, context_length, device)
            logits = model(inputs)
            total_loss += cross_entropy(logits, targets).item()
    model.train()
    return total_loss / eval_iters


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer language model")

    # Data
    parser.add_argument("--train_data", type=str, required=True, help="Path to tokenized training data (uint16 .npy or raw binary)")
    parser.add_argument("--val_data", type=str, required=True, help="Path to tokenized validation data")

    # Model
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Optimizer
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--eps", type=float, default=1e-8)

    # Logging / checkpointing
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")

    # Misc
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name; omit to disable logging")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device

    # --- Data (memory-mapped for large files) ---
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")

    # --- Model ---
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # --- Optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # --- Resume ---
    start_iter = 0
    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from checkpoint at iteration {start_iter}")

    # --- Checkpoint dir ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- W&B ---
    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, config=vars(args))

    # --- Training loop ---
    model.train()
    t0 = time.time()

    for iteration in range(start_iter, args.max_iters):
        # LR schedule
        lr = lr_cosine_schedule(iteration, args.max_lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(list(model.parameters()), args.max_grad_norm)
        optimizer.step()

        # Logging
        if iteration % args.log_interval == 0:
            elapsed = time.time() - t0
            print(f"iter {iteration:>6d} | loss {loss.item():.4f} | lr {lr:.6f} | {elapsed:.1f}s")
            if args.wandb_project is not None:
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/iter": iteration})

        # Evaluation
        if iteration % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
            print(f"  >>> val loss: {val_loss:.4f}")
            if args.wandb_project is not None:
                wandb.log({"val/loss": val_loss, "train/iter": iteration})

        # Checkpointing
        if iteration > 0 and iteration % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, ckpt_path)
            print(f"  >>> saved checkpoint to {ckpt_path}")

    # Final checkpoint
    ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_final_{args.max_iters}.pt")
    save_checkpoint(model, optimizer, args.max_iters, ckpt_path)
    print(f"Training complete. Final checkpoint saved to {ckpt_path}")

    if args.wandb_project is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
