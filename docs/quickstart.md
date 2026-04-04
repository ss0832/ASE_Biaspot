# Quickstart Guide

ASE_Biaspot lets you apply **user-defined bias potentials** on top of any ASE
geometry optimization. If PyTorch is available, forces are computed via autograd;
otherwise the library falls back to finite differences (FD) automatically.

---

## Installation

```bash
# Install from PyPI
pip install ase-biaspot
```

### Development install from source

```bash
# Core install 
pip install -e ".[dev]"
```

---

## Example 1 — Simplest usage (distance restraint)

Pull the H–H bond of H₂ toward 0.5 Å (shorter than the EMT equilibrium of 0.779 Å).

```python
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from ase_biaspot import BiasTerm, BiasCalculator

atoms = molecule("H2")
atoms.calc = EMT()

# Bias function: f(vars_, params) -> float (eV)
def bias_fn(vars_, params):
    r  = vars_["r"]
    k  = params["k"]
    r0 = params["r0"]
    return k * (r - r0) ** 2

# Build a BiasTerm from a callable
term = BiasTerm.from_callable(
    name="harmonic_distance",
    fn=bias_fn,
    variables={"r": lambda ctx: ctx.distance(0, 1)},  # geometry extractor
    params={"k": 5.0, "r0": 0.5},
)

biased = BiasCalculator(
    base_calculator=EMT(),
    terms=[term],
    gradient_mode="auto",  # torch autograd if available, else FD
    verbose=True,           # print energy/force summary each step
)
atoms.calc = biased

opt = BFGS(atoms, logfile=None)
opt.run(fmax=0.05, steps=60)

print(f"H-H distance: {atoms.get_distance(0, 1):.4f} Å")
# => H-H distance: 0.7536 Å  (shifted from EMT equilibrium 0.779 Å)
```

**Example output with `verbose=True`:**

```
[ASE_Biaspot] step=1 E_base=1.15886 E_bias=0.281239 E_total=1.44010 Fmax=2.09024 terms: harmonic_distance=0.281239
...
[ASE_Biaspot] step=5 E_base=1.10172 E_bias=0.321552 E_total=1.42327 Fmax=0.000497 terms: harmonic_distance=0.321552
```

> **Note:** `verbose=True` prints `E_base` (base calculator), `E_bias` (per-term),
> `E_total`, and `Fmax` at every step.

---

## Example 2 — Dict spec with multiple terms

`term_from_spec()` lets you define bias terms declaratively as dictionaries —
no need to write a callable by hand.

The example below applies an **angle restraint** (HOH → 90°) and an **AFIR term**
(push H–H together) to a water molecule simultaneously.

```python
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import LBFGS

from ase_biaspot import BiasCalculator, term_from_spec

atoms = molecule("H2O")
atoms.calc = EMT()

# Angle restraint: pull HOH angle toward 90°
# Use the "expression" key (string) — this spec is fully JSON/YAML-serialisable.
#
# ⚠️  Name uniqueness: variables and params are merged into one namespace
#     for expression evaluation.  If a key appears in BOTH dicts a ValueError
#     is raised at evaluation time — use distinct names to avoid this.
#     e.g. variable "th" and param "th0" are fine; naming both "th" is not.
angle_spec = {
    "name": "angle_bias",
    "type": "expression_callable",
    "expression": "k * (th - th0) ** 2",  # "th" (variable) and "k","th0" (params) — no overlap
    "variables": {
        "th": {"type": "angle", "atoms": [1, 0, 2], "unit": "deg"}
    },
    "params": {"k": 0.05, "th0": 90.0},
}

# AFIR term: push H(index 1) and H(index 2) together
afir_spec = {
    "name": "afir_hh",
    "type": "afir",
    "params": {
        "gamma": 1.5,    # kJ/mol, positive → push together
        "group_a": [1],
        "group_b": [2],
    },
}

terms = [term_from_spec(angle_spec), term_from_spec(afir_spec)]

biased = BiasCalculator(
    base_calculator=EMT(),
    terms=terms,
    gradient_mode="auto",
    verbose=False,
    csv_log_path="h2o_run.csv",  # save per-step log to CSV
)
atoms.calc = biased

opt = LBFGS(atoms, logfile=None)
opt.run(fmax=0.05, steps=80)

print(f"HOH angle: {atoms.get_angle(1, 0, 2):.2f}°")
print(f"H-H distance: {atoms.get_distance(1, 2):.4f} Å")
```

**Built-in variable types:**

| `type` | `atoms` key meaning | `unit` |
|---|---|---|
| `"distance"` | `[i, j]` | — (Å) |
| `"angle"` | `[i, j, k]` — j is the vertex | `"deg"` / `"rad"` |
| `"dihedral"` | `[i, j, k, l]` | `"deg"` / `"rad"` |
| `"out_of_plane"` | `[i, j, k, l]` — i is the out-of-plane atom | `"deg"` / `"rad"` |

**Built-in `type` names:**

| `type` | Description | JSON/YAML-safe? |
|---|---|---|
| `"afir"` | AFIR artificial-force bias (Maeda–Morokuma) | ✅ |
| `"expression_callable"` with `"expression"` key | String expression bias (e.g. `"k*(r-r0)**2"`) | ✅ |
| `"expression_callable"` with `"callable"` key | Python callable passed directly | ❌ |
| `"callable"` | Python callable passed directly (fully supported; not deprecated) | ❌ |
| `"torch_callable"` | Torch-native callable with learnable `nn.Parameter` weights | ❌ |
| `"torch_afir"` | AFIR term with learnable `gamma` (`nn.Parameter`) | ⚠️ spec のみ |

> **Note — `"torch_afir"` の JSON/YAML-safe 欄について**
> `"torch_afir"` の spec dict 自体は primitive 値のみで構成されるため、JSON/YAML への
> シリアライズは可能です。ただし、**学習済みの `gamma` 値 (`nn.Parameter`) は spec
> に含まれない**ため、JSON からロードすると `gamma` は初期値にリセットされます。
> 学習済みパラメータを永続化するには `torch.save(term.state_dict(), path)` を
> 別途使用してください。

> **Note — `expression_callable` naming constraint**
> When using the `"expression"` string form, variable names (`variables` dict)
> and parameter names (`params` dict) are merged into one namespace before the
> expression is evaluated.  A key that appears in **both** dicts raises a
> `ValueError` at evaluation time with a message listing the conflicting names.
> Keep all names distinct — e.g. use `r` for a distance variable and `r0` for
> its equilibrium parameter, never the same string for both.

---

## Example 3 — Learnable parameters (TorchCallableTerm)

Store the bias spring constant as an `nn.Parameter` and update it with an
external optimizer after each geometry step. `∂E_bias/∂k` is computed
automatically.

```python
import torch
import torch.nn as nn
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from ase_biaspot import BiasCalculator, TorchCallableTerm

atoms = molecule("H2")

# Learnable spring constant k (initial value 1.0)
k_param = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

term = TorchCallableTerm(
    name="learnable_harmonic",
    fn=lambda v, p: p["k"] * (v["r"] - p["r0"]) ** 2,
    variables={"r": lambda ctx: ctx.distance(0, 1)},
    fixed_params={"r0": 0.60},        # non-learnable constant
    trainable_params={"k": k_param},  # learnable parameter
)

# External optimizer that updates k
param_opt = torch.optim.Adam(term.parameters(), lr=0.2)

biased = BiasCalculator(
    base_calculator=EMT(),
    terms=[term],
    gradient_mode="auto",
    verbose=False,
    zero_param_grads=True,  # reset param.grad before each backward pass
)
atoms.calc = biased

def observer():
    """Update k with Adam after each BFGS step."""
    r_hh  = atoms.get_distance(0, 1)
    k_val = k_param.item()
    k_g   = k_param.grad.item() if k_param.grad is not None else float("nan")
    print(f"  r={r_hh:.3f} Å  k={k_val:.4f}  ∂E/∂k={k_g:.5f}")

    param_opt.step()
    param_opt.zero_grad()

opt = BFGS(atoms, logfile=None)
opt.attach(observer)
opt.run(fmax=0.05, steps=30)
```

> **Use case:** Automatically tune bias strength to match experimental reference
> values or target trajectories (e.g. force-field fitting, reaction-path guiding).

---

## Example 4 — Custom BiasTerm subclass

Subclass `BiasTerm` directly to implement a completely custom potential.

```python
import numpy as np
from ase_biaspot import BiasTerm, BiasCalculator
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS

class RepulsiveTerm(BiasTerm):
    """Soft-core repulsion: E = epsilon * (sigma / r) ^ 12"""

    def __init__(self, name: str, i: int, j: int,
                 epsilon: float, sigma: float):
        self.name    = name
        self.i       = i
        self.j       = j
        self.epsilon = epsilon
        self.sigma   = sigma

    def evaluate(self, positions, atomic_numbers=None):
        r = np.linalg.norm(positions[self.i] - positions[self.j])
        return self.epsilon * (self.sigma / r) ** 12

atoms = molecule("H2")
term  = RepulsiveTerm("lj_repulsion", i=0, j=1, epsilon=0.5, sigma=0.8)

biased = BiasCalculator(
    base_calculator=EMT(),
    terms=[term],
    gradient_mode="fd",   # NumPy implementation → use finite differences
)
atoms.calc = biased

opt = BFGS(atoms, logfile=None)
opt.run(fmax=0.05, steps=50)

print(f"H-H distance: {atoms.get_distance(0, 1):.4f} Å")
# => ~0.84 Å  (wider than EMT equilibrium due to repulsion)
```

**Subclassing contract:**

| Method / attribute | Required? | Description |
|---|---|---|
| `self.name` (set in `__init__`) | **Yes** | String identifier; not setting it raises `TypeError` immediately |
| `evaluate(positions, atomic_numbers)` | **Yes** | Returns energy (eV) as a NumPy float |
| `supports_autograd` property | Optional | Return `True` to enable autograd path |
| `evaluate_tensor(positions, atomic_numbers)` | When autograd enabled | Returns energy as a Torch scalar tensor |

> **`self.name` is enforced at instantiation time.** If a subclass `__init__`
> does not assign `self.name`, a `TypeError` is raised immediately when the
> object is created — not later when `BiasCalculator.calculate()` is called.
> Dataclass-based subclasses (like `AFIRTerm`) are unaffected because their
> generated `__init__` always sets `self.name`.

> **Important — `evaluate_tensor()` must return a scalar tensor (shape=()).**
> If your function naturally produces a vector (e.g. per-pair contributions),
> reduce it explicitly before returning: `return contributions.sum()`.
> Returning a non-scalar raises `TypeError` in `BiasCalculator`.

## Energy units

The value returned by `evaluate()` and `evaluate_tensor()` must always be in
**eV**. `BiasCalculator` applies no unit conversion internally.

If you accidentally implement a term in kJ/mol, forces will be off by a factor
of ~96. To make the mistake visible, declare `energy_unit` on your subclass —
`BiasCalculator` will emit a `UserWarning` at construction time:

```python
class MyKJTerm(CallableTerm):
    energy_unit = "kJ/mol"  # → UserWarning emitted by BiasCalculator
```

Setting `energy_unit = "eV"` (the default) produces no warning.

---

## Registering a custom dict type

To make a custom term available to `term_from_spec()`, decorate a builder
function with `@register`:

```python
from ase_biaspot.factory import register
from ase_biaspot import BiasTerm

@register("my_repulsion")
def _build_repulsion(name: str, spec: dict) -> BiasTerm:
    p = spec["params"]
    return RepulsiveTerm(
        name=name,
        i=p["i"], j=p["j"],
        epsilon=p["epsilon"], sigma=p["sigma"],
    )

# Now usable via dict spec:
spec = {
    "name": "my_rep",
    "type": "my_repulsion",
    "params": {"i": 0, "j": 1, "epsilon": 0.5, "sigma": 0.8},
}
term = term_from_spec(spec)
```

---

## gradient_mode reference

| `gradient_mode` | Behaviour |
|---|---|
| `"auto"` *(default)* | torch autograd if PyTorch available, else FD |
| `"torch"` | Force autograd; raises `ImportError` if PyTorch is missing |
| `"fd"` | Finite differences for all terms |

> **Note:** `gradient_mode` is case-insensitive — `"FD"`, `"Auto"`, `"TORCH"` etc. are all accepted and normalised to lowercase internally.

- Terms inheriting from `TorchBiasTerm` (`nn.Module`) always use autograd
  regardless of this setting.
- Terms with `supports_autograd = False` (e.g. `CallableTerm`) always use FD
  silently — no warning is emitted.
- When `gradient_mode="fd"` is set explicitly, no warning is emitted even for
  autograd-capable terms.  The warning is only emitted on automatic fallback
  (i.e. `gradient_mode="auto"` and PyTorch is not installed).

---

## Logging

```python
biased = BiasCalculator(
    base_calculator=EMT(),
    terms=[term],
    verbose=True,             # print per-step summary to stdout
    csv_log_path="run.csv",   # also save to CSV (None to disable)
)
```

**CSV columns:**

```
step, E_base, E_bias_total, E_total, Fmax, bias_<term_name>, ...
```

---

## Example 6 — Custom BiasTerm with autograd: Morse potential

`BiasTerm` をサブクラス化し、`evaluate_tensor()` を実装することで
**torch autograd に対応したカスタムポテンシャル**を定義できます。

以下の `MorseBiasTerm` は Morse ポテンシャルをバイアス項として実装した例です。

$$E_{\rm Morse} = D_e \bigl(1 - e^{-a(r - r_e)}\bigr)^2$$

調和ポテンシャルと異なり、$r \to \infty$ で $E \to D_e$ に漸近するため、
大変位域でも力が発散しません。

```python
import math
import numpy as np
import torch

from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from ase_biaspot import BiasTerm, BiasCalculator


class MorseBiasTerm(BiasTerm):
    """
    Morse ポテンシャルバイアス項。

    E = D_e * (1 - exp(-a * (r - r_e)))^2

    Parameters
    ----------
    name : str
        ログ出力に使われる識別子。
    i, j : int
        距離を計算する原子インデックス (0-based)。
    D_e : float
        解離エネルギー [eV]。ポテンシャルの上限値。
    a : float
        範囲パラメータ [1/Å]。大きいほど井戸が狭い。
    r_e : float
        ポテンシャル最小点の原子間距離 [Å]。
    """

    def __init__(self, name: str, i: int, j: int,
                 D_e: float, a: float, r_e: float) -> None:
        self.name = name
        self.i, self.j = i, j
        self.D_e, self.a, self.r_e = D_e, a, r_e

    @property
    def supports_autograd(self) -> bool:
        return True  # evaluate_tensor() を実装しているので True

    def evaluate(self, positions, atomic_numbers=None) -> float:
        """NumPy パス (FD 勾配の基礎)。"""
        r = np.linalg.norm(positions[self.i] - positions[self.j])
        return float(self.D_e * (1.0 - math.exp(-self.a * (r - self.r_e))) ** 2)

    def evaluate_tensor(self, positions, atomic_numbers=None):
        """Torch autograd パス。計算グラフを保持したままエネルギーを返す。"""
        diff = positions[self.i] - positions[self.j]
        r = torch.linalg.norm(diff)
        return self.D_e * (1.0 - torch.exp(-self.a * (r - self.r_e))) ** 2


atoms = molecule("H2")

morse = MorseBiasTerm(
    name="morse_hh",
    i=0, j=1,
    D_e=1.0,   # eV
    a=3.0,     # 1/Å
    r_e=1.1,   # Å — EMT 平衡(0.74 Å)より長い目標距離
)

biased = BiasCalculator(
    base_calculator=EMT(),
    terms=[morse],
    gradient_mode="auto",  # torch autograd を使用
    verbose=True,
)
atoms.calc = biased

opt = BFGS(atoms, logfile=None)
opt.run(fmax=0.05, steps=60)

print(f"H-H distance: {atoms.get_distance(0, 1):.4f} Å")
# => H-H distance: 0.9228 Å  (EMT 平衡 0.74 Å → Morse 最小点 1.1 Å の間で収束)
```

**Subclassing contract との対応:**

| 要件 | 対応箇所 |
|---|---|
| `self.name` を `__init__` で設定 | `self.name = name` |
| `evaluate()` の実装 (NumPy, FD パス) | `np.linalg.norm` + `math.exp` |
| `supports_autograd = True` を返す | `@property` で `True` |
| `evaluate_tensor()` の実装 (Torch, autograd パス) | `torch.linalg.norm` + `torch.exp` |

> **Note:** `evaluate_tensor()` は必ず rank-0 テンソル (shape=`()`) を返す必要があります。
> ベクトルを返すと `BiasCalculator` が `TypeError` を送出します。
> 複数ペアへの寄与を合計する場合は `contributions.sum()` で明示的に縮約してください。

---

## Example 7 — Gaussian soft-wall between fragment groups

2 つの原子グループの**重心間距離**に作用するガウス型反発ポテンシャルです。

$$E_{\rm Gauss} = A \cdot \exp\!\left(-\frac{r_{\rm cm}^2}{2\sigma^2}\right)$$

フラグメント同士が接近しすぎることを柔らかく防ぐソフトウォールとして機能します。
$A < 0$ にすると引力バイアスにもなります。

```python
import numpy as np
import torch

from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from ase_biaspot import BiasTerm, BiasCalculator


class GaussianRepulsionTerm(BiasTerm):
    """
    重心間距離に作用するガウス型反発ポテンシャル。

    E = A * exp(-r_cm^2 / (2 * sigma^2))

    Parameters
    ----------
    name : str
        識別子。
    group_a, group_b : list[int]
        2 つのフラグメントの原子インデックスリスト。
    A : float
        ガウスの振幅 [eV]。正 → 反発、負 → 引力。
    sigma : float
        ガウスの幅 [Å]。
    """

    def __init__(self, name: str, group_a: list[int], group_b: list[int],
                 A: float, sigma: float) -> None:
        self.name = name
        self.group_a = list(group_a)
        self.group_b = list(group_b)
        self.A, self.sigma = A, sigma

    @property
    def supports_autograd(self) -> bool:
        return True

    def evaluate(self, positions, atomic_numbers=None) -> float:
        com_a = positions[self.group_a].mean(axis=0)
        com_b = positions[self.group_b].mean(axis=0)
        r = float(np.linalg.norm(com_a - com_b))
        return float(self.A * np.exp(-(r**2) / (2.0 * self.sigma**2)))

    def evaluate_tensor(self, positions, atomic_numbers=None):
        # positions の mean を通じて計算グラフが保持される
        com_a = positions[self.group_a].mean(dim=0)
        com_b = positions[self.group_b].mean(dim=0)
        r = torch.linalg.norm(com_a - com_b)
        return self.A * torch.exp(-(r**2) / (2.0 * self.sigma**2))


# 2 つの H₂ を近距離に配置
atoms = Atoms(
    "H4",
    positions=[
        (0.0, 0.0, 0.0),
        (0.7, 0.0, 0.0),
        (1.5, 0.0, 0.0),  # 重心間距離 1.5 Å — 近接
        (2.2, 0.0, 0.0),
    ],
)

gauss_wall = GaussianRepulsionTerm(
    name="gauss_wall",
    group_a=[0, 1],
    group_b=[2, 3],
    A=2.0,      # eV
    sigma=1.5,  # Å
)

biased = BiasCalculator(
    base_calculator=EMT(),
    terms=[gauss_wall],
    gradient_mode="auto",
    verbose=True,
)
atoms.calc = biased

opt = BFGS(atoms, logfile=None)
opt.run(fmax=0.05, steps=60)

r_cm = np.linalg.norm(
    atoms.positions[[0, 1]].mean(axis=0) - atoms.positions[[2, 3]].mean(axis=0)
)
print(f"重心間距離: {r_cm:.4f} Å")
# => 重心間距離: 5.5579 Å  (ガウス反発によって 1.5 Å から押し広げられた)
```

**複数のバイアス項を組み合わせる例 (Morse + Gaussian):**

```python
from ase.build import molecule
from ase.optimize import LBFGS

atoms = molecule("H2O")

# O-H 結合を 1.1 Å に引き伸ばす Morse バイアス (両結合)
morse_oh1 = MorseBiasTerm(name="morse_oh1", i=0, j=1, D_e=0.8, a=2.5, r_e=1.1)
morse_oh2 = MorseBiasTerm(name="morse_oh2", i=0, j=2, D_e=0.8, a=2.5, r_e=1.1)

# H-H 近接を防ぐガウス反発
gauss_hh = GaussianRepulsionTerm(
    name="gauss_hh", group_a=[1], group_b=[2], A=0.5, sigma=0.8
)

biased = BiasCalculator(
    base_calculator=EMT(),
    terms=[morse_oh1, morse_oh2, gauss_hh],
    gradient_mode="auto",
    verbose=False,
)
atoms.calc = biased

opt = LBFGS(atoms, logfile=None)
opt.run(fmax=0.05, steps=80)

print(f"O-H(1): {atoms.get_distance(0, 1):.4f} Å")  # => 1.0992 Å (目標 1.1 Å)
print(f"O-H(2): {atoms.get_distance(0, 2):.4f} Å")  # => 1.0992 Å
print(f"HOH:    {atoms.get_angle(1, 0, 2):.2f}°")   # => 137.77°
```

---

## Quick-reference table

| Goal | Class / function |
|---|---|
| Write bias as a lambda / def | `BiasTerm.from_callable()` |
| Declare bias as a dict | `term_from_spec()` |
| Apply AFIR artificial force | `AFIRTerm` / dict `type: "afir"` |
| Fully custom term | `BiasTerm` subclass |
| Custom term with autograd | `BiasTerm` subclass + `evaluate_tensor()` + `supports_autograd = True` |
| Morse-type bond bias | `MorseBiasTerm` (Example 6) |
| Soft-wall / Gaussian repulsion between groups | `GaussianRepulsionTerm` (Example 7) |
| Learnable parameters | `TorchCallableTerm` / `TorchAFIRTerm` |
| Register a new dict type | `@register("my_type")` |

---

## Example 5 — Claisen rearrangement TS search with Psi4 + AFIR

This example shows how to use ASE_Biaspot with a real quantum-chemistry
calculator ([Psi4](https://psicode.org/)) to search for the transition state
of an allyl vinyl ether **Claisen rearrangement** ([3,3]-sigmatropic).

The AFIR term pushes the reacting fragments together, nudging the system
away from the reactant minimum and toward the TS region without requiring
an initial guess for the product geometry.

### Prerequisites

```bash
#After installing psi4
pip install ase-biaspot   # with torch for autograd forces
```

### Step 1 — AFIR-biased geometry optimization

```python
# claisen_afir.py
#
# AFIR-driven geometry optimisation for the Claisen rearrangement of
# allyl vinyl ether → pent-4-enal  ([3,3] sigmatropic shift).
#
# 0-based atom index map
# ──────────────────────
#  idx  sym  role in Claisen 6-membered TS
#   0    C   C2  vinyl carbon attached to O  (=CH–O)
#   1    O   O3  ether oxygen
#   2    C   C4  allyl methylene attached to O  (–CH₂–)   ← breaking bond (1–2)
#   3    C   C5  allyl middle carbon  (=CH–)
#   4    C   C6  allyl terminal carbon  (=CH₂)             ← forming bond (4–5)
#   5    C   C1  vinyl terminal carbon  (=CH₂)             ← forming bond (4–5)
#  6–13  H   hydrogens
#
# Claisen [3,3] key bonds
# ───────────────────────
#   forming : C1 (idx 5) ··· C6 (idx 4)   new C–C bond (reactant d ≈ 4.84 Å)
#   breaking: O3 (idx 1) –– C4 (idx 2)   C–O bond cleaves (reactant d ≈ 1.43 Å)

import numpy as np
from ase import Atoms
from ase.calculators.psi4 import Psi4
from ase.io import write
from ase.optimize import LBFGS

from ase_biaspot import AFIRTerm, BiasCalculator

# ── Build the allyl vinyl ether reactant from coordinates ────────────────────

symbols = [
    "C", "O", "C", "C", "C", "C",          # 0–5  heavy atoms
    "H", "H", "H", "H", "H", "H", "H", "H",  # 6–13 hydrogens
]
positions = np.array([
    [-0.03989051, -1.44876143, -4.10172098],  #  0: C2  vinyl (–O side)
    [-0.24675417, -0.16352192, -4.69355444],  #  1: O3  ether oxygen
    [ 0.99900999,  0.34386140, -5.17887270],  #  2: C4  allyl methylene (–O side)
    [ 1.99215654,  0.47134851, -4.00882699],  #  3: C5  allyl middle
    [ 3.32183313,  0.33200893, -4.23043860],  #  4: C6  allyl terminal  ← new bond
    [-0.83456106, -1.86233332, -3.08485125],  #  5: C1  vinyl terminal  ← new bond
    [ 0.74232880, -2.08390788, -4.46175144],  #  6: H
    [-0.67977551, -2.82401638, -2.64201141],  #  7: H
    [-1.61677881, -1.22718586, -2.72481917],  #  8: H
    [ 1.39589792, -0.32674955, -5.91212635],  #  9: H
    [ 0.84422389,  1.30554411, -5.62171311],  # 10: H
    [ 1.63235197,  0.66994296, -3.02089940],  # 11: H
    [ 3.68163775,  0.13341526, -5.21836632],  # 12: H
    [ 4.01187671,  0.42059024, -3.41748521],  # 13: H
])

atoms = Atoms(symbols=symbols, positions=positions)

# ── QM calculator: B3LYP/6-31G* via Psi4 ────────────────────────────────────

qm_calc = Psi4(
    atoms=atoms,
    method="b3lyp",
    basis="6-31g*",
    memory="4GB",
    num_threads=4,
)

# ── AFIR bias term ────────────────────────────────────────────────────────────
#
# group_a — allyl fragment   : C4, C5, C6  (indices 2, 3, 4)
# group_b — vinyl ether part : C1, C2, O3  (indices 5, 0, 1)
#
# gamma = 150 kJ/mol pushes the two fragments together along the reaction
# coordinate.  Increase to 200–250 kJ/mol if the system does not reach
# the TS vicinity within 300 steps.

afir_term = AFIRTerm(
    name="claisen_afir",
    group_a=[2, 3, 4],   # allyl fragment  (C4–CH₂, C5=CH–, C6=CH₂)
    group_b=[0, 1, 5],   # vinyl ether fragment  (C2=CH–, O3, C1=CH₂)
    gamma=150.0,          # kJ/mol — positive: push together
    power=6.0,
)

biased = BiasCalculator(
    base_calculator=qm_calc,
    terms=[afir_term],
    gradient_mode="auto",   # torch autograd if available, else FD
    verbose=True,
    csv_log_path="claisen_afir.csv",
)
atoms.calc = biased

# ── Optimisation ──────────────────────────────────────────────────────────────

opt = LBFGS(atoms, trajectory="claisen_afir.traj", logfile="claisen_afir.log")
opt.run(fmax=0.05, steps=300)

# ── Save AFIR-optimised structure (TS vicinity / approximate product) ─────────

write("claisen_afir_product.xyz", atoms)
print("AFIR optimisation complete.")
```

> **Tip — choosing gamma:**
> Start with γ = 100 kJ/mol.  If the reaction does not occur within ~200 steps,
> increase to 150–200 kJ/mol.  Very large values (> 400 kJ/mol) can distort bond
> lengths significantly; use them only as a last resort.

### Step 2 — Locate the transition state with NEB

Convert the AFIR trajectory to a set of images and run a climbing-image NEB
to locate the true TS between the reactant and the AFIR product.

> **Converting trajectories:** To extract individual frames for NEB from an
> ASE `.traj` file use:
> ```bash
> ase convert claisen_afir.traj claisen_afir.xyz
> ```

### References

See the {doc}`references` page for the AFIR, ASE, Psi4, and NEB citations
relevant to this workflow.
