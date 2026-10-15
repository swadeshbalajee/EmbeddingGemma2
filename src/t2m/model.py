import torch
from torch import nn
from torch.nn import functional as F


class PrefixAdapter(nn.Module):
    """Map one unit-norm 768d vector into one or more LM prefix tokens."""
    def __init__(self, in_dim, lm_dim, hidden=1024, initial_prefix_norm=1.0, num_tokens=1):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.lm_dim = int(lm_dim)
        if self.num_tokens < 1:
            raise ValueError("num_tokens must be positive")
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, self.num_tokens * self.lm_dim))
        self.scale = nn.Parameter(torch.tensor(float(initial_prefix_norm)))

    def forward(self, z):
        p = self.net(z.float()).reshape(z.shape[0], self.num_tokens, self.lm_dim)
        # Match mean native token-embedding norm at initialization; trainable scale.
        return F.normalize(p, dim=-1) * self.scale.abs().clamp(min=1e-3)


class PrefixConditionedLM(nn.Module):
    def __init__(self, lm, embedding_dim=768, adapter_hidden=1024, num_prefix_tokens=1):
        super().__init__()
        self.lm = lm
        dim = lm.get_input_embeddings().weight.shape[1]
        with torch.no_grad():
            w = lm.get_input_embeddings().weight
            norm = w.detach().float().norm(dim=-1).mean().item()
        self.adapter = PrefixAdapter(embedding_dim, dim, adapter_hidden, norm,
                                     num_tokens=num_prefix_tokens)
        self.num_prefix_tokens = int(num_prefix_tokens)

    def forward(self, embeddings, input_ids, labels, mask, negative_indices,
                rank_lambda=0.2, rank_margin=0.2, noise_std=0.0):
        return self.loss_and_metrics(embeddings, input_ids, labels, mask, negative_indices,
                                     rank_lambda, rank_margin, noise_std)

    def prefix(self, embedding, noise_std=0.0):
        z = F.normalize(embedding.float(), dim=-1)
        if self.training and noise_std:
            z = F.normalize(z + torch.randn_like(z) * noise_std, dim=-1)
        return self.adapter(z)

    def token_logprobs(self, embeddings, input_ids, labels, mask, noise_std=0.0):
        """input_ids=[BOS]+targets[:-1]; logits at BOS predict first target.

        The prefix occupies positions 0..K-1; BOS is position K. Built-in label shifting
        is deliberately NOT used to prevent off-by-one and prefix supervision.
        """
        p = self.prefix(embeddings, noise_std)
        tok = self.lm.get_input_embeddings()(input_ids)
        x = torch.cat((p.to(tok.dtype), tok), dim=1)
        prefix_mask = torch.ones((mask.shape[0], self.num_prefix_tokens), device=mask.device, dtype=mask.dtype)
        full_mask = torch.cat((prefix_mask, mask), dim=1)
        logits = self.lm(inputs_embeds=x, attention_mask=full_mask, use_cache=False).logits
        # Predictions at position K .. K+T-1 correspond to labels 0 .. T-1.
        target_logits = logits[:, self.num_prefix_tokens:, :]
        t = labels.shape[1]
        loss_tokens = F.cross_entropy(
            target_logits.float().reshape(-1, target_logits.shape[-1]),
            labels.reshape(-1), reduction="none", ignore_index=-100,
        ).view(labels.shape[0], t)
        valid = labels.ne(-100)
        counts = valid.sum(dim=1).clamp(min=1)
        score = -loss_tokens.sum(dim=1) / counts
        return score, loss_tokens, counts

    def loss_and_metrics(self, embeddings, input_ids, labels, mask, negative_indices,
                         rank_lambda=0.2, rank_margin=0.2, noise_std=0.0):
        b = embeddings.shape[0]
        if b < 2:
            raise ValueError("At least 2 distinct training records per GPU microbatch required")
        # Single batched forward for positive and negative contexts; identical token histories.
        embs = torch.cat((embeddings, embeddings[negative_indices]), dim=0)
        inputs = torch.cat((input_ids, input_ids), dim=0)
        ys = torch.cat((labels, labels), dim=0)
        masks = torch.cat((mask, mask), dim=0)
        score, token_nll, count = self.token_logprobs(embs, inputs, ys, masks, noise_std)
        pos, neg = score[:b], score[b:]
        delta = pos - neg
        ce = -pos.mean()
        rank = F.softplus(rank_margin - delta).mean()
        total = ce + rank_lambda * rank
        # Tokens 1-8 (excluding EOS by position, not content), and rest; diagnostic only.
        early = token_nll[:b, :8].sum() / count[:b].clamp(max=8).sum().clamp(min=1)
        late_count = (count[:b] - 8).clamp(min=0)
        late = token_nll[:b, 8:].sum() / late_count.sum().clamp(min=1)
        metrics = {
            "ce": ce.detach(), "rank_loss": rank.detach(), "loss": total.detach(),
            "delta_cond": delta.mean().detach(),
            "rank_acc": (delta > 0).float().mean().detach(),
            "margin_acc": (delta >= rank_margin).float().mean().detach(),
            "pos_nll": (-pos).mean().detach(), "neg_nll": (-neg).mean().detach(),
            "early_nll": early.detach(), "late_nll": late.detach(),
            "prefix_norm": self.adapter.scale.abs().detach(),
        }
        return total, metrics

    @torch.no_grad()
    def generate_from_embedding(self, z, bos_token_id, eos_token_id, pad_token_id,
                                max_new_tokens=96, repetition_penalty=1.08):
        """Greedy token decoding. Uses model KV caching after initial [prefix, BOS]."""
        was_training = self.training
        self.eval()
        try:
            device = z.device
            b = z.shape[0]
            bos = torch.full((b, 1), bos_token_id, device=device, dtype=torch.long)
            pref = self.prefix(z)
            token = self.lm.get_input_embeddings()(bos)
            inputs = torch.cat((pref.to(token.dtype), token), dim=1)
            gen = []
            done = torch.zeros(b, device=device, dtype=torch.bool)
            attention_mask = torch.ones((b, self.num_prefix_tokens + 1), device=device, dtype=torch.long)
            out = self.lm(inputs_embeds=inputs, attention_mask=attention_mask, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1].float()
            for _ in range(max_new_tokens):
                if repetition_penalty != 1.0 and gen:
                    for i, seq in enumerate(torch.stack(gen, dim=1).tolist()):
                        used = set(seq)
                        for token_id in used:
                            if token_id != pad_token_id:
                                score = logits[i, token_id]
                                logits[i, token_id] = score * repetition_penalty if score < 0 else score / repetition_penalty
                next_t = logits.argmax(dim=-1)
                next_t = torch.where(done, torch.full_like(next_t, pad_token_id), next_t)
                gen.append(next_t)
                done |= (next_t == eos_token_id)
                if done.all():
                    break
                attention_mask = torch.cat((attention_mask, torch.ones((b, 1), device=device, dtype=torch.long)), 1)
                out = self.lm(input_ids=next_t.unsqueeze(1), attention_mask=attention_mask,
                              past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1].float()
            return torch.stack(gen, dim=1) if gen else torch.empty((b, 0), dtype=torch.long, device=device)
        finally:
            self.train(was_training)
