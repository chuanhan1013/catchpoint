"""Re-implementation of the unitree_rl_gym G1 recurrent actor (LSTM 47->64, MLP 64->32->12) so it can be fine-tuned."""
import os
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY = os.path.join(HERE, "..", "third_party", "unitree_rl_gym", "deploy", "pre_train", "g1", "motion.pt")


class Actor(nn.Module):
    def __init__(self, act=nn.ELU):
        super().__init__()
        self.memory = nn.LSTM(47, 64)
        self.actor = nn.Sequential(nn.Linear(64, 32), act(), nn.Linear(32, 12))

    def forward(self, x, hc):
        """x (T, B, 47), hc = (h, c) each (1, B, 64) -> mu (T, B, 12), hc"""
        out, hc = self.memory(x, hc)
        return self.actor(out), hc

    def zero_state(self, B):
        z = torch.zeros(1, B, 64)
        return (z, z.clone())


def load_pretrained():
    sm = torch.jit.load(POLICY)
    sd = {k: v for k, v in sm.state_dict().items() if not k.endswith(("hidden_state", "cell_state"))}
    a1 = getattr(sm.actor, "1")
    a = Actor()
    a.load_state_dict(sd)
    return a, sm, a1.original_name


def export_script(actor, path):
    """TorchScript module with the same stateful single-step interface as motion.pt."""
    class Stateful(nn.Module):
        def __init__(self, a):
            super().__init__()
            self.memory, self.actor = a.memory, a.actor
            self.register_buffer("hidden_state", torch.zeros(1, 1, 64))
            self.register_buffer("cell_state", torch.zeros(1, 1, 64))

        def forward(self, x):
            out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
            self.hidden_state.copy_(h)
            self.cell_state.copy_(c)
            return self.actor(out.squeeze(0))
    m = torch.jit.script(Stateful(actor).eval())
    m.save(path)
    return path


if __name__ == "__main__":
    a, sm, act_name = load_pretrained()
    print("activation in pretrained:", act_name)
    torch.manual_seed(0)
    xs = torch.randn(30, 1, 47)
    hc = a.zero_state(1)
    mu, _ = a(xs, hc)
    ref = torch.stack([sm(xs[t]) for t in range(30)])
    print("max diff vs scripted:", (mu - ref).abs().max().item())
