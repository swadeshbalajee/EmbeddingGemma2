import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config
from .data import CachedDataset, LengthBucketBatchSampler, collate, different_indices
from .eval import evaluate_text, evaluate_media
from .model import PrefixConditionedLM


def init_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        if world > 1:
            raise RuntimeError("Multi-rank training requires CUDA in this repo")
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    return rank, world, device


def check_cache_metadata(ds, cfg):
    m = ds.meta
    for key, value in (("encoder_id",cfg["encoder_id"]),("decoder_id",cfg["decoder_id"]),
                       ("max_seq_len",cfg["max_seq_len"]),("embedding_dim",768)):
        if m.get(key) != value:
            raise RuntimeError(f"Cached {key}={m.get(key)} differs from config {value}; rebuild the cache")


def wrap(raw, device, world):
    if world > 1:
        return DDP(raw, device_ids=[device.index], output_device=device.index,
                   broadcast_buffers=False, find_unused_parameters=False,
                   gradient_as_bucket_view=True)
    return raw


def cosine_schedule(optimizer, total_steps, warmup_steps):
    def factor(s):
        if s < warmup_steps:
            return max(1e-3,(s+1)/max(1,warmup_steps))
        progress = (s-warmup_steps)/max(1,total_steps-warmup_steps)
        return max(.05, .5*(1+math.cos(math.pi*min(1.,progress))))
    return LambdaLR(optimizer, factor)


def make_optimizer(raw, cfg, phase, total_steps):
    param_groups = [{"params":list(raw.adapter.parameters()), "lr":float(cfg["lr_adapter"])}]
    if phase == "joint":
        param_groups.append({"params":[p for p in raw.lm.parameters() if p.requires_grad],
                             "lr":float(cfg["lr_decoder"])})
    fused = bool(torch.cuda.is_available())
    optimizer = AdamW(param_groups, weight_decay=float(cfg["weight_decay"]), fused=fused)
    schedule = cosine_schedule(optimizer,total_steps,max(1,round(total_steps*float(cfg["warmup_ratio"]))))
    return optimizer, schedule


def save_checkpoint(path, raw, optimizer, scheduler, step, epoch, next_batch, phase, cfg,
                    include_optimizer=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Optional compiled LM has _orig_mod. in state_dict; save canonical names.
    state = {k.replace("lm._orig_mod.","lm."):v.detach().cpu() for k,v in raw.state_dict().items()}
    package = {"model":state,"step":step,"epoch":epoch,"next_batch":next_batch,
               "phase":phase,"config":cfg,"torch_rng":torch.get_rng_state()}
    if include_optimizer:
        package["optimizer"] = optimizer.state_dict()
        package["scheduler"] = scheduler.state_dict()
    temp = path.with_suffix(".tmp")
    torch.save(package,temp)
    temp.replace(path)


def make_model(cfg, device):
    tok = AutoTokenizer.from_pretrained(cfg["decoder_id"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if tok.bos_token_id is None and tok.eos_token_id is None:
        raise ValueError("Need BOS or EOS token for generation")
    lm = AutoModelForCausalLM.from_pretrained(cfg["decoder_id"],
         attn_implementation=cfg.get("attn_implementation","sdpa"), torch_dtype=torch.float32)
    if hasattr(lm.config, "use_cache"):
        lm.config.use_cache = False
    if cfg.get("gradient_checkpointing", False):
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    else:
        lm.gradient_checkpointing_disable()
    raw = PrefixConditionedLM(lm, adapter_hidden=int(cfg["adapter_hidden"]),
                              num_prefix_tokens=int(cfg.get("num_prefix_tokens", 1))).to(device)
    if cfg.get("torch_compile", False):
        if not hasattr(torch,"compile"):
            raise RuntimeError("torch.compile unsupported; disable torch_compile")
        raw.lm = torch.compile(raw.lm, mode=cfg.get("compile_mode","default"),dynamic=True)
    return raw, tok


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--config", default="configs/v1b.yaml")
    a.add_argument("--skip-media-eval", action="store_true", help="Only for pipeline debugging")
    args = a.parse_args()
    cfg = load_config(args.config)
    rank, world, device = init_distributed()
    random.seed(int(cfg["seed"])+rank)
    np.random.seed(int(cfg["seed"])+rank)
    torch.manual_seed(int(cfg["seed"])+rank)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("tf32",True))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True,exist_ok=True)
    if rank == 0:
        (outdir/"config_resolved.json").write_text(json.dumps(cfg,indent=2))
    if not args.skip_media_eval and not (Path(cfg["eval_cache"])/"image_embeddings.npy").exists():
        raise RuntimeError("Missing image evaluation cache. Run scripts/prepare_demo_images.py and "
                           "python -m t2m.cache_media --kind image --input-dir data/eval_images "
                           "--out data/cache/eval_media")
    if world > 1:
        dist.barrier()
    train_ds = CachedDataset(cfg["train_cache"])
    val_ds = CachedDataset(cfg["val_cache"])
    check_cache_metadata(train_ds,cfg)
    check_cache_metadata(val_ds,cfg)
    raw, tok = make_model(cfg,device)
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pad_id = tok.pad_token_id
    if int(cfg["batch_size_per_gpu"]) < 2:
        raise ValueError("batch_size_per_gpu >= 2 required for in-batch negative ranking")
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True,drop_last=True)
    loader_kwargs = {"num_workers": int(cfg["num_workers"]), "pin_memory": device.type=="cuda",
                     "persistent_workers": int(cfg["num_workers"])>0,
                     "collate_fn": lambda rows: collate(rows,bos_id,pad_id)}
    if cfg.get("length_bucket_size"):
        batch_sampler = LengthBucketBatchSampler(
            sampler, train_ds.lengths, int(cfg["batch_size_per_gpu"]),
            bucket_size=int(cfg["length_bucket_size"]), seed=int(cfg["seed"]))
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        loader = DataLoader(train_ds, batch_size=int(cfg["batch_size_per_gpu"]),
                            sampler=sampler, drop_last=True, **loader_kwargs)
    accum = int(cfg["grad_accum_steps"])
    steps_per_epoch = max(1,len(loader)//accum)
    total_steps = int(cfg.get("max_steps") or 0) or int(cfg["epochs"])*steps_per_epoch
    initial_warm = min(int(cfg["adapter_warmup_steps"]),max(total_steps-1,0))
    step, start_epoch, start_batch = 0, 0, 0
    package = None
    if cfg.get("resume"):
        package = torch.load(cfg["resume"],map_location="cpu",weights_only=False)
        state = package["model"]
        if cfg.get("torch_compile", False):
            state = {k.replace("lm.","lm._orig_mod.",1) if k.startswith("lm.") else k:v for k,v in state.items()}
        raw.load_state_dict(state)
        step,start_epoch,start_batch = package["step"],package["epoch"],package["next_batch"]
    warmup = step < initial_warm
    raw.lm.requires_grad_(not warmup)
    model = wrap(raw, device, world)
    phase = "adapter" if warmup else "joint"
    optimizer, scheduler = make_optimizer(raw, cfg, phase, max(1,initial_warm if warmup else total_steps-initial_warm))
    if package:
        if package["phase"] != phase:
            raise ValueError("Checkpoint training phase mismatch")
        if "optimizer" not in package or "scheduler" not in package:
            raise ValueError("This model-only checkpoint cannot resume training")
        optimizer.load_state_dict(package["optimizer"])
        scheduler.load_state_dict(package["scheduler"])
    wb = None
    if rank == 0 and cfg.get("wandb"):
        import wandb
        wb = wandb.init(project=cfg.get("wandb_project","text2modal-v1"),
                        name=f'{cfg["variant"]}-{outdir.name}',config=cfg,resume="allow")
    logpath = outdir/"metrics.jsonl"
    t0 = time.monotonic()
    seen_tokens = 0
    epoch = start_epoch
    last_print = time.monotonic()
    while epoch < int(cfg["epochs"]) and step < total_steps:
        sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        raw.train()
        micro_in_accum = 0
        progress = tqdm(loader, total=len(loader), desc=f'{cfg["variant"]} epoch {epoch+1}/{cfg["epochs"]}',
                        unit="batch", disable=rank != 0, dynamic_ncols=True, mininterval=1)
        for batch_idx, batch in enumerate(progress):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            if step >= total_steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k,v in batch.items()}
            z, tokens, labels, mask = (batch[k] for k in ("embeddings","input_ids","labels","mask"))
            perm = different_indices(len(z),device)
            # Do NOT turn off model grads: prefix gradients flow through frozen LM during warmup.
            syncing = (micro_in_accum+1)%accum==0
            sync_context = contextlib.nullcontext() if syncing or world==1 else model.no_sync()
            with sync_context:
                with torch.autocast("cuda",dtype=torch.bfloat16, enabled=device.type=="cuda" and torch.cuda.is_bf16_supported()):
                    # DDP needs its forward() to track reducer state.
                    loss, metrics = model(z,tokens,labels,mask,perm,
                                         float(cfg["rank_lambda"]),float(cfg["rank_margin"]),
                                         float(cfg["noise_std"]))
                (loss/accum).backward()
            seen_tokens += int(labels.ne(-100).sum()) * 2
            micro_in_accum += 1
            if not syncing:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in raw.parameters() if p.requires_grad],float(cfg["grad_clip"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            micro_in_accum = 0
            wall = max(1e-9,time.monotonic()-t0)
            rec = {"step":step,"epoch":epoch,"phase":phase,"variant":cfg["variant"],
                   "tokens_per_sec_total":seen_tokens/wall,"grad_norm":float(grad_norm),
                   "lr_adapter":float(optimizer.param_groups[0]["lr"]),
                   "gpu_allocated_gb":torch.cuda.max_memory_allocated(device)/1e9 if device.type=="cuda" else 0.,
                   **{f"train/{k}":float(v) for k,v in metrics.items()}}
            if rank==0 and step % int(cfg["log_every"]) == 0:
                with logpath.open("a") as f:
                    f.write(json.dumps(rec)+"\n")
                if wb: wb.log(rec,step=step)
                print(json.dumps(rec),flush=True)
            text_eval = step % int(cfg["text_eval_every"])==0
            media_eval = step % int(cfg["media_eval_every"])==0
            # Checkpoints exactly at the adapter->joint transition are deferred
            # because the optimizer/phase changes immediately afterward.
            save = (step % int(cfg["checkpoint_every"])==0
                    and not (phase=="adapter" and step>=initial_warm))
            if text_eval or media_eval or save:
                if world>1: dist.barrier()
                if rank==0:
                    if text_eval:
                        val = evaluate_text(raw,val_ds,bos_id,pad_id,device,
                                     batch_size=int(cfg["batch_size_per_gpu"]),n_batches=int(cfg["text_eval_batches"]))
                        vrec = {"step":step,"epoch":epoch,"variant":cfg["variant"],
                                **{f"val/{k}":v for k,v in val.items()}}
                        with logpath.open("a") as f: f.write(json.dumps(vrec)+"\n")
                        if wb: wb.log(vrec,step=step)
                        print(json.dumps(vrec),flush=True)
                    if media_eval and not args.skip_media_eval:
                        path, n = evaluate_media(raw,tok,cfg["eval_cache"],outdir,step,cfg["variant"],
                                   device,limit=int(cfg["media_eval_limit"]),
                                   max_new_tokens=int(cfg["max_new_tokens"]),
                                   repetition_penalty=float(cfg["repetition_penalty"]))
                        print(f"Media preview {path}: {n} rows",flush=True)
                    if save:
                        # Resume starts with the next batch index (checkpoints after updates).
                        save_checkpoint(outdir/"checkpoints"/f"step_{step:07d}.pt",raw,
                                       optimizer,scheduler,step,epoch,batch_idx+1,phase,cfg)
                        keep = int(cfg.get("checkpoint_keep_last", 0))
                        if keep > 0:
                            old = sorted((outdir / "checkpoints").glob("step_*.pt"))[:-keep]
                            for stale in old:
                                stale.unlink()
                    raw.train()
                if world>1: dist.barrier()
            if phase=="adapter" and step>=initial_warm:
                if world>1: dist.barrier()
                del model
                raw.lm.requires_grad_(True)
                model=wrap(raw,device,world)
                phase="joint"
                optimizer,scheduler=make_optimizer(raw,cfg,phase,max(1,total_steps-step))
                if rank==0: print(f"Transition to joint fine-tuning at step {step}",flush=True)
                if world>1: dist.barrier()
        start_batch=0
        epoch+=1
    if rank==0:
        save_checkpoint(outdir/"checkpoints"/"final.pt",raw,optimizer,scheduler,
                        step,epoch,0,phase,cfg,
                        include_optimizer=bool(cfg.get("save_optimizer_final", True)))
        if cfg.get("cleanup_checkpoints_after_final", False):
            for stale in (outdir / "checkpoints").glob("step_*.pt"):
                stale.unlink()
        if wb: wb.finish()
        print(f"Training complete. Total optimizer steps: {step}")
    if world>1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
