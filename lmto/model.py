import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmto import config
from lmto.data import DIMS

DEV = "cuda"
MAX_GRID = 192


class LMTO(nn.Module):
    """A pretrained causal LM over [text header | image and point tokens | blocks]. A block of 4 x 4 elements enters
    through a linear map and leaves through 16 logits, at learned embeddings of its row and column. The generator
    reads the blocks autoregressively from a start vector; the corrector reads one query per block, which also embeds
    the forced elements of its block, and attends over all of them at once."""

    def __init__(self, base, dims, corrector=False):
        super().__init__()
        self.base = base
        d = self.d = base.config.hidden_size
        emb = base.get_input_embeddings().weight.detach().float()
        mean, std = emb.mean(0), emb.std()
        self.start = nn.Parameter(mean.clone())
        self.query = nn.Parameter(mean.clone())
        self.row = nn.Embedding(MAX_GRID, d)
        self.col = nn.Embedding(MAX_GRID, d)
        for e in (self.row, self.col):
            nn.init.normal_(e.weight, std=float(std) / 2 ** 0.5)
        self.img_in = nn.ModuleDict({k: nn.Linear(n, d) for k, n in dims.items()})
        self.unit_in, self.unit_out = nn.Linear(16, d), nn.Linear(d, 16)
        nn.init.zeros_(self.unit_out.weight)
        nn.init.zeros_(self.unit_out.bias)
        self.force_in = None
        if corrector:
            self.force_in = nn.Linear(16, d)
            nn.init.zeros_(self.force_in.weight)
            nn.init.zeros_(self.force_in.bias)

    def embed_text(self, ids):
        return self.base.get_input_embeddings()(ids)

    def pos_embed(self, rc):
        return self.row(rc[..., 0].clamp(max=MAX_GRID - 1)) + self.col(rc[..., 1].clamp(max=MAX_GRID - 1))

    def unit_embed(self, rc, forced):
        e = self.pos_embed(rc)
        return e if self.force_in is None else e + self.force_in(forced)

    def head(self, h):
        return self.unit_out(h)

    def backbone(self, embeds, mask, pos):
        return self.base.model(inputs_embeds=embeds, attention_mask=mask, position_ids=pos).last_hidden_state


def block_mask(prefix, n, total, bidirectional, dtype, device):
    """Additive (B, 1, L, L) mask: causal over the prefix, causal or bidirectional over the blocks, padding blocked."""
    i = torch.arange(total, device=device)
    allow = (i[None, :] <= i[:, None]).expand(len(prefix), total, total).clone()
    for b in range(len(prefix)):
        s, e = prefix[b], prefix[b] + n[b]
        if bidirectional:
            allow[b, s:e, s:e] = True
        allow[b, :, e:] = False
        allow[b, e:, :] = False
        allow[b, e:, e:] = torch.eye(total - e, dtype=torch.bool, device=device)
    m = torch.zeros(len(prefix), 1, total, total, dtype=dtype, device=device)
    return m.masked_fill_(~allow[:, None], torch.finfo(dtype).min)


def build(cfg):
    """The pretrained LM in bfloat16 with LoRA on all linear layers, and the new layers in float32."""
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16)
    lora = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear", lora_dropout=0.0)
    base = get_peft_model(base, lora).base_model.model
    torch.manual_seed(1234)
    corrector = cfg.get("corrector", False)
    dims = {k: n for k, n in DIMS.items() if corrector or k != "draft"}
    return LMTO(base, dims, corrector).to(DEV), tok


def load(name):
    cfg = config.CONFIGS[name]
    model, tok = build(cfg)
    state = torch.load(f"{config.RESULTS}/{name}/weights.pt", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected[:5]
    return cfg, model.eval(), tok


def prefix_embed(model, it):
    parts = [model.embed_text(torch.tensor(it["ids"], device=DEV))]
    for name, x, rc in it["imgs"]:
        x = model.img_in[name](torch.from_numpy(x).to(DEV)) + model.pos_embed(torch.from_numpy(rc).to(DEV))
        parts.append(x.to(parts[0].dtype))
    return torch.cat(parts, 0)
