# variations/triadic_modulation_variations.py
import torch
import torch.nn.functional as F

# ── Biologically-motivated MOD (context-sensitive) transfer functions ──────────
def cooperation(r, c):
    """
    f(R,C) = ReLU6( R/2 + 2R + C*(1+|R|) )
    Shapes: r, c  ->  (..., E)
    """
    return F.relu6(r / 2.0 + 2 * r + c * (1 + torch.abs(r)))

def tm1(r, c):
    return 0.5 * r * (1 + torch.exp(r * c))

def tm2(r, c):
    return r + r * c

def tm3(r, c):
    return r * (1 + torch.tanh(r * c))

def tm4(r, c):
    return r * (2 * r * c)

mod_fn_dict = {
    "cooperation": cooperation,
    "tm1": tm1,
    "tm2": tm2,
    "tm3": tm3,
    "tm4": tm4,
}

