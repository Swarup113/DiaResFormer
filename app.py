"""
app.py  —  ResFormer Diabetes Diagnostic Flask App
===================================================
XAI methods:
  1. Permutation Feature Importance  (pre-computed, loaded from pkl)
  2. ALE – Accumulated Local Effects  (pre-computed, loaded from pkl)
  3. Attention Rollout                (computed per-request, ~10 ms)
  4. Counterfactual Explanations      (computed per-request, gradient-free)

Medically-backed input validation is applied before inference.

Run locally:
    pip install flask torch numpy scikit-learn matplotlib
    python app.py

Deploy (Railway / Hugging Face Spaces):
    gunicorn app:app --workers 1 --threads 4 --timeout 120
"""

import os, math, io, base64, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template

import torch
import torch.nn as nn

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR  = os.path.join(BASE_DIR, "xai_artifacts")
DEVICE   = torch.device("cpu")

# ══════════════════════════════════════════════════════════════════════════════
# MEDICALLY-BACKED INPUT VALIDATION RANGES
# ══════════════════════════════════════════════════════════════════════════════
#
# Pregnancies  0–20
#   The highest reliably recorded number of pregnancies is ~17-19 in clinical
#   literature; 20 is a safe absolute ceiling.
#
# Glucose (2-hour OGTT plasma glucose)  50–300 mg/dL
#   Below 50 mg/dL = severe hypoglycaemia (emergency, not an outpatient value).
#   Above 300 mg/dL = extreme uncontrolled hyperglycaemia incompatible with a
#   routine diagnostic visit.
#
# Blood Pressure (diastolic)  40–130 mmHg
#   Diastolic BP < 40 mmHg indicates haemodynamic shock.
#   Diastolic BP > 130 mmHg is a hypertensive emergency.
#
# Skin Thickness (triceps skinfold)  5–99 mm
#   Values below 5 mm are physiologically implausible in adult women.
#   No published clinical measurement exceeds ~99 mm.
#
# Insulin (2-hour serum insulin)  2–846 uU/mL
#   Fasting insulin below 2 uU/mL suggests severe beta-cell failure.
#   846 uU/mL is the dataset maximum and a recognised clinical upper bound.
#
# BMI  15.0–70.0 kg/m²
#   WHO lowest documented category starts at 16 (severe thinness); 15 accepted
#   as an absolute floor. Upper bound 70 kg/m² covers extreme obesity as
#   documented in peer-reviewed case reports.
#
# Diabetes Pedigree Function  0.078–3.0
#   Original Pima dataset range is 0.078–2.42; 3.0 accepted as ceiling.
#
# Age  21–120 years
#   The Pima dataset covers women aged 21+. 120 is the verified human lifespan
#   maximum.
#
MEDICAL_RANGES = {
    # key               : (min,    max,   display_name,           unit    )
    "pregnancies"       : (0,      20,    "Pregnancies",          "count" ),
    "glucose"           : (50,     300,   "Glucose",              "mg/dL" ),
    "blood_pressure"    : (40,     130,   "Blood Pressure",       "mmHg"  ),
    "skin_thickness"    : (5,      99,    "Skin Thickness",       "mm"    ),
    "insulin"           : (2,      846,   "Insulin",              "uU/mL" ),
    "bmi"               : (15.0,   70.0,  "BMI",                  "kg/m2" ),
    "dpf"               : (0.078,  3.0,   "Diabetes Pedigree Fn", "score" ),
    "age"               : (21,     120,   "Age",                  "years" ),
}


def validate_ranges(values: dict):
    """
    Check every feature against MEDICAL_RANGES.
    Returns an error string on the first failure, or None if all pass.
    """
    for key, (lo, hi, label, unit) in MEDICAL_RANGES.items():
        val = values.get(key)
        if val is None:
            return f"{label} is required."
        if not (lo <= val <= hi):
            return (
                f"{label} value {val} {unit} is outside the accepted clinical "
                f"range ({lo}–{hi} {unit}). Please check your entry."
            )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION  (must be byte-for-byte identical to the training script)
# ══════════════════════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class ResBlock(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(d), nn.ReLU(),
            nn.Linear(d, d, bias=False), nn.Dropout(dropout),
            nn.BatchNorm1d(d), nn.ReLU(),
            nn.Linear(d, d, bias=False), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResFormer(nn.Module):
    def __init__(self, n_features: int, d_token: int = 128, n_blocks_res: int = 2,
                 n_blocks_trans: int = 2, n_heads: int = 4, ffn_mult: int = 4,
                 dropout_res: float = 0.1, dropout_attn: float = 0.1,
                 n_classes: int = 2):
        super().__init__()
        self.n_features = n_features
        self.d_token    = d_token
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.res_input_proj = nn.Sequential(
            nn.Linear(n_features * d_token, d_token),
            nn.BatchNorm1d(d_token),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(d_token, dropout_res) for _ in range(n_blocks_res)]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=d_token * ffn_mult,
            dropout=dropout_attn, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=n_blocks_trans)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_tokens = self.tokenizer(x)
        flat = feat_tokens.reshape(x.size(0), self.n_features * self.d_token)
        cls  = self.res_input_proj(flat)
        cls  = self.res_blocks(cls)
        seq  = torch.cat([cls.unsqueeze(1), feat_tokens], dim=1)
        seq  = self.transformer(seq)
        return self.head(seq[:, 0, :])

    def predict_proba(self, X_np: np.ndarray) -> np.ndarray:
        """numpy (N, F)  →  numpy (N, C) softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self(torch.tensor(X_np, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def get_attention_rollout(self, x_tensor: torch.Tensor) -> np.ndarray:
        """
        Attention Rollout (Abnar & Zuidema, 2020).
        Accumulates attention across all Transformer layers:
            R_l = normalise(A_l + I) · R_{l-1},   R_0 = I
        Returns a normalised [0,1] per-feature importance vector (n_features,).
        """
        self.eval()
        feat_tokens = self.tokenizer(x_tensor)
        flat = feat_tokens.reshape(x_tensor.size(0),
                                   self.n_features * self.d_token)
        cls  = self.res_input_proj(flat)
        cls  = self.res_blocks(cls)
        seq  = torch.cat([cls.unsqueeze(1), feat_tokens], dim=1)

        T       = seq.size(1)       # CLS + F feature tokens
        rollout = torch.eye(T)      # identity initialisation

        for layer in self.transformer.layers:
            seq_norm = layer.norm1(seq)
            _, attn  = layer.self_attn(
                seq_norm, seq_norm, seq_norm,
                need_weights=True, average_attn_weights=True,
            )
            attn_avg = attn.mean(dim=0).detach().cpu()      # (T, T)
            attn_aug = attn_avg + torch.eye(T)
            attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
            rollout  = torch.matmul(attn_aug, rollout)
            seq      = layer(seq)

        feat_imp = rollout[0, 1:].numpy()   # CLS row, skip CLS→CLS
        feat_imp = feat_imp - feat_imp.min()
        if feat_imp.max() > 0:
            feat_imp /= feat_imp.max()
        return feat_imp


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

def load_artifacts():
    ckpt = torch.load(
        os.path.join(ART_DIR, "resformer_best.pt"),
        map_location=DEVICE,
        weights_only=False,   # checkpoint contains numpy scalars → must be False
    )
    model = ResFormer(**ckpt["model_config"]).to(DEVICE)
    model.load_state_dict(
        {k: v.to(DEVICE) for k, v in ckpt["model_state_dict"].items()}
    )
    model.eval()

    def _pkl(name):
        with open(os.path.join(ART_DIR, name), "rb") as fh:
            return pickle.load(fh)

    return (
        model,
        _pkl("scaler.pkl"),
        _pkl("feature_names.pkl"),
        _pkl("class_info.pkl"),
        _pkl("pfi_values.pkl"),
        _pkl("ale_data.pkl"),
        _pkl("cf_config.pkl"),
    )


print("Loading ResFormer model and XAI artifacts …")
MODEL, SCALER, FEAT_NAMES, CLASS_INFO, PFI, ALE_DATA, CF_CFG = load_artifacts()
N_FEAT = len(FEAT_NAMES)
print(f"  Ready | features: {FEAT_NAMES}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT PALETTE  — white/green to match DiaResFormer frontend
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "bg"     : "#f8fafc",   # site background (slate-50)
    "card"   : "#ffffff",   # card white
    "green"  : "#10b981",   # primary emerald-500
    "green2" : "#34d399",   # emerald-400 (lighter fill)
    "red"    : "#ef4444",   # red-500
    "text"   : "#0f172a",   # slate-900
    "sub"    : "#64748b",   # slate-500 (muted labels)
    "border" : "#e2e8f0",   # slate-200
}


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _style_ax(ax, title: str = ""):
    """Apply consistent white-background styling to an Axes."""
    ax.set_facecolor(PALETTE["card"])
    ax.tick_params(colors=PALETTE["sub"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["sub"])
    ax.yaxis.label.set_color(PALETTE["sub"])
    if title:
        ax.set_title(title, fontweight="bold", pad=8,
                     color=PALETTE["text"], fontsize=10)
    else:
        ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    ax.grid(visible=False)   # no grid on white background


# ══════════════════════════════════════════════════════════════════════════════
# XAI PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Permutation Feature Importance ────────────────────────────────────────
def plot_pfi() -> str:
    """
    Horizontal bar chart of pre-computed PFI values (global explanation).
    Green = importance > 0 (loss increases when feature is shuffled).
    Red   = near-zero or negative (feature has little effect on loss).
    """
    vals         = np.array([PFI[f] for f in FEAT_NAMES])
    order        = np.argsort(vals)[::-1]          # descending
    feats_sorted = [FEAT_NAMES[i] for i in order]
    vals_sorted  = vals[order]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    fig.patch.set_facecolor(PALETTE["bg"])

    bar_colors = [PALETTE["green"] if v >= 0 else PALETTE["red"]
                  for v in vals_sorted]

    ax.barh(
        range(N_FEAT), vals_sorted[::-1],
        color=bar_colors[::-1],
        edgecolor=PALETTE["border"], linewidth=0.5, height=0.65,
    )
    ax.set_yticks(range(N_FEAT))
    ax.set_yticklabels(feats_sorted[::-1], fontsize=9, color=PALETTE["text"])
    ax.set_xlabel("Mean Δ Cross-Entropy Loss  (↑ = more important)",
                  color=PALETTE["sub"], fontsize=8)
    ax.axvline(0, color=PALETTE["sub"], lw=0.8, ls="--")

    max_abs = max(abs(vals_sorted).max(), 1e-9)
    for i, v in enumerate(vals_sorted[::-1]):
        ax.text(
            v + max_abs * 0.02 if v >= 0 else v - max_abs * 0.02,
            i, f"{v:+.4f}",
            va="center", ha="left" if v >= 0 else "right",
            fontsize=7.5, color=PALETTE["sub"],
        )

    _style_ax(ax, "Permutation Feature Importance")
    plt.tight_layout()
    return _fig_to_b64(fig)


# ── 2. ALE — all features grid ───────────────────────────────────────────────
def plot_ale_all(x_scaled: np.ndarray) -> str:
    """
    2×4 grid of ALE curves for all 8 features.
    Green shading = ALE > 0 (risk-increasing region).
    Red shading   = ALE < 0 (risk-decreasing region).
    Dotted vertical line = user's own value.
    """
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        "Accumulated Local Effects — All Features",
        color=PALETTE["text"], fontsize=12, fontweight="bold", y=1.02,
    )

    for idx, fname in enumerate(FEAT_NAMES):
        ax   = axes[idx // ncols][idx % ncols]
        data = ALE_DATA.get(fname)
        if data is None:
            ax.axis("off")
            continue

        grid = data["grid"]
        ale  = data["ale"]
        fi   = FEAT_NAMES.index(fname)

        # Inverse-transform grid to original units for x-axis labels
        dummy_g         = np.zeros((len(grid), N_FEAT))
        dummy_g[:, fi]  = grid
        orig_grid       = SCALER.inverse_transform(dummy_g)[:, fi]

        ax.plot(orig_grid, ale, color=PALETTE["green"], lw=2, zorder=3)
        ax.fill_between(orig_grid, ale, 0,
                        where=(ale >= 0), alpha=0.20, color=PALETTE["green2"])
        ax.fill_between(orig_grid, ale, 0,
                        where=(ale < 0),  alpha=0.20, color=PALETTE["red"])
        ax.axhline(0, color=PALETTE["sub"], lw=0.7, ls="--")

        # Mark user's value
        dummy_u         = np.zeros((1, N_FEAT))
        dummy_u[0, fi]  = x_scaled[0, fi]
        user_orig       = float(SCALER.inverse_transform(dummy_u)[0, fi])
        ax.axvline(user_orig, color=PALETTE["green"], lw=1.4, ls=":",
                   label=f"You: {user_orig:.1f}")
        ax.legend(fontsize=7, loc="upper right",
                  facecolor=PALETTE["card"], edgecolor=PALETTE["border"],
                  labelcolor=PALETTE["text"])

        ax.set_xlabel("Feature value", fontsize=7, color=PALETTE["sub"])
        ax.set_ylabel("ALE", fontsize=7, color=PALETTE["sub"])
        _style_ax(ax, fname)

    plt.tight_layout()
    return _fig_to_b64(fig)


# ── 3. Attention Rollout ──────────────────────────────────────────────────────
def plot_attention_rollout(x_scaled: np.ndarray) -> str:
    """
    Horizontal bar chart of per-feature Attention Rollout weights [0,1].
    YlGn colormap (yellow-green) reads cleanly on white.
    Value labels on each bar.
    """
    x_t          = torch.tensor(x_scaled, dtype=torch.float32)
    rolls        = MODEL.get_attention_rollout(x_t)   # (N_FEAT,) in [0,1]
    order        = np.argsort(rolls)[::-1]
    feats_sorted = [FEAT_NAMES[i] for i in order]
    vals_sorted  = rolls[order]

    cmap   = plt.cm.get_cmap("YlGn")
    # Offset the colormap to avoid near-white bars at low values
    colors = [cmap(0.35 + 0.55 * v) for v in vals_sorted]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax.barh(
        range(N_FEAT), vals_sorted[::-1],
        color=colors[::-1],
        edgecolor=PALETTE["border"], linewidth=0.4, height=0.65,
    )
    ax.set_yticks(range(N_FEAT))
    ax.set_yticklabels(feats_sorted[::-1], fontsize=9, color=PALETTE["text"])
    ax.set_xlabel("Normalised Attention Rollout Weight  [0 – 1]",
                  color=PALETTE["sub"], fontsize=8)
    ax.set_xlim(0, 1.12)

    for i, v in enumerate(vals_sorted[::-1]):
        ax.text(v + 0.012, i, f"{v:.3f}",
                va="center", fontsize=7.5, color=PALETTE["sub"])

    _style_ax(ax, "Attention Rollout — Per-Input Feature Focus")
    plt.tight_layout()
    return _fig_to_b64(fig)


# ── 4. Counterfactual Explanations ────────────────────────────────────────────
def counterfactual_explanation(
    x_orig_unscaled: np.ndarray,
    target_class: int = 0,
    max_iter: int = 1000,
    step_scale: float = 0.3,
) -> dict:
    """
    Probability-guided hill-climbing counterfactual search.

    Each iteration shuffles the mutable features and tries both +step and
    -step for each.  The move that most increases P(target_class) is accepted.
    If a full pass over all features yields no improvement the step size grows
    by 50% to escape flat plateaus.

    target_class = 0  →  search for Non-Diabetic outcome  (pred was 1)
    target_class = 1  →  search for Diabetic outcome      (pred was 0)

    Pregnancies is locked (historical, cannot be changed by lifestyle).
    """
    feat_names = CF_CFG["feature_names"]
    feat_min   = np.array(CF_CFG["feat_min"],  dtype=np.float64)
    feat_max   = np.array(CF_CFG["feat_max"],  dtype=np.float64)
    feat_step  = np.array(CF_CFG["feat_step"], dtype=np.float64) * step_scale
    int_feats  = CF_CFG["integer_feats"]

    mutable = [i for i, f in enumerate(feat_names) if f != "Pregnancies"]

    rng      = np.random.default_rng(0)
    best_cf  = x_orig_unscaled.copy().astype(np.float64)
    found    = False

    def prob_target(x: np.ndarray) -> float:
        xs = SCALER.transform(x.reshape(1, -1)).astype(np.float32)
        return float(MODEL.predict_proba(xs)[0][target_class])

    best_prob = prob_target(best_cf)

    for _iter in range(max_iter):
        feat_order = rng.permutation(mutable)
        improved   = False

        for fi in feat_order:
            best_dir_x    = None
            best_dir_prob = best_prob

            for direction in (-1.0, 1.0):
                x_try     = best_cf.copy()
                x_try[fi] = np.clip(
                    best_cf[fi] + direction * feat_step[fi],
                    feat_min[fi], feat_max[fi],
                )
                if feat_names[fi] in int_feats:
                    x_try[fi] = round(x_try[fi])

                p = prob_target(x_try)
                if p > best_dir_prob:
                    best_dir_prob = p
                    best_dir_x    = x_try.copy()

            if best_dir_x is not None:
                best_cf   = best_dir_x
                best_prob = best_dir_prob
                improved  = True

            if best_prob > 0.5:
                found = True
                break   # inner loop; outer 'if found: break' handles full exit

        if found:
            break

        if not improved:                          # plateau escape
            feat_step = np.minimum(
                feat_step * 1.5,
                (feat_max - feat_min) * 0.1,
            )

    x_s_cf   = SCALER.transform(best_cf.reshape(1, -1)).astype(np.float32)
    cf_proba = MODEL.predict_proba(x_s_cf)[0].tolist()

    changes = []
    for i, fn in enumerate(feat_names):
        orig = float(x_orig_unscaled[i])
        cf_v = float(best_cf[i])
        if abs(orig - cf_v) > 1e-4:
            changes.append({
                "feature"        : fn,
                "original"       : round(orig, 2),
                "counterfactual" : round(cf_v, 2),
                "delta"          : round(cf_v - orig, 2),
            })

    return {
        "found"    : found,
        "changes"  : changes,
        "cf_proba" : cf_proba,
        "target"   : CF_CFG["class_names"][target_class],
        "cf_array" : best_cf.tolist(),
    }


def plot_counterfactual(x_orig: np.ndarray, cf_result: dict) -> str:
    """
    Horizontal diverging bar chart of required feature changes (Δ).
    Green = decrease needed (health-improving).
    Red   = increase needed.
    Each bar annotated with 'original → counterfactual'.
    """
    changes = cf_result["changes"]
    if not changes:
        return ""

    feats  = [c["feature"]        for c in changes]
    orig_v = [c["original"]       for c in changes]
    cf_v   = [c["counterfactual"] for c in changes]
    deltas = [c["delta"]          for c in changes]

    y_pos  = np.arange(len(feats))
    colors = [PALETTE["green"] if d < 0 else PALETTE["red"] for d in deltas]

    fig, ax = plt.subplots(figsize=(7, max(3.8, len(feats) * 0.6 + 1.8)))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax.barh(y_pos, deltas, color=colors,
            edgecolor=PALETTE["border"], linewidth=0.5, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=9, color=PALETTE["text"])
    ax.set_xlabel("Required change  (Δ)", color=PALETTE["sub"], fontsize=8)
    ax.axvline(0, color=PALETTE["sub"], lw=0.9, ls="--")

    max_abs = max((abs(d) for d in deltas), default=1.0)
    for i, (o, c, d) in enumerate(zip(orig_v, cf_v, deltas)):
        pad = max_abs * 0.02
        ax.text(
            pad if d >= 0 else -pad, i,
            f"  {o:.1f} → {c:.1f}",
            va="center", ha="left" if d >= 0 else "right",
            fontsize=8, color=PALETTE["text"],
        )

    target_label = cf_result.get("target", "target class")
    _style_ax(
        ax,
        f"Counterfactual Explanation\n(changes needed for {target_label} outcome)",
    )
    plt.tight_layout()
    return _fig_to_b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    ─────────────
    JSON body fields:
      pregnancies, glucose, blood_pressure, skin_thickness,
      insulin, dpf, age
      PLUS either:
        bmi                       (direct)
      OR:
        weight_kg + height_cm     (auto-calculated)

    Returns prediction, confidence, bmi_calculated,
    and xai dict with 4 base64 PNG images + text summaries.
    """
    data = request.get_json(force=True)

    # ── 1. Parse ─────────────────────────────────────────────────────────────
    try:
        pregnancies    = float(data["pregnancies"])
        glucose        = float(data["glucose"])
        blood_pressure = float(data["blood_pressure"])
        skin_thickness = float(data["skin_thickness"])
        insulin        = float(data["insulin"])
        dpf            = float(data["dpf"])
        age            = float(data["age"])

        bmi_raw = data.get("bmi")
        if bmi_raw not in (None, "", "0", 0):
            bmi = float(bmi_raw)
        else:
            weight_kg = float(data["weight_kg"])
            height_cm = float(data["height_cm"])
            if height_cm <= 0:
                return jsonify({"error": "Height must be greater than 0 cm."}), 400
            if weight_kg <= 0:
                return jsonify({"error": "Weight must be greater than 0 kg."}), 400
            bmi = weight_kg / ((height_cm / 100.0) ** 2)

    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid or missing field: {exc}"}), 400

    # ── 2. Medical range validation ───────────────────────────────────────────
    range_error = validate_ranges({
        "pregnancies"    : pregnancies,
        "glucose"        : glucose,
        "blood_pressure" : blood_pressure,
        "skin_thickness" : skin_thickness,
        "insulin"        : insulin,
        "bmi"            : bmi,
        "dpf"            : dpf,
        "age"            : age,
    })
    if range_error:
        return jsonify({"error": range_error}), 400

    # ── 3. Build feature vector ───────────────────────────────────────────────
    # Order MUST match training:
    # Pregnancies, Glucose, BloodPressure, SkinThickness,
    # Insulin, BMI, DiabetesPedigreeFunction, Age
    x_raw    = np.array([[pregnancies, glucose, blood_pressure,
                           skin_thickness, insulin, bmi, dpf, age]],
                         dtype=np.float32)
    x_scaled = SCALER.transform(x_raw).astype(np.float32)

    # ── 4. Inference ──────────────────────────────────────────────────────────
    proba       = MODEL.predict_proba(x_scaled)[0]   # [P(Non-Diabetic), P(Diabetic)]
    pred        = int(np.argmax(proba))
    class_names = ["Non-Diabetic", "Diabetic"]

    confidence = {
        "non_diabetic": round(float(proba[0]) * 100, 2),
        "diabetic"    : round(float(proba[1]) * 100, 2),
    }

    # ── 5. XAI ────────────────────────────────────────────────────────────────

    # (a) PFI — global, pre-computed, identical for every request
    pfi_img = plot_pfi()

    # (b) ALE — pre-computed curves + per-request user value marker
    ale_img = plot_ale_all(x_scaled)

    # (c) Attention Rollout — ~10 ms on CPU
    attn_img = plot_attention_rollout(x_scaled)

    # (d) Counterfactual — target is the opposite of current prediction
    target_cls = 0 if pred == 1 else 1
    cf_result  = counterfactual_explanation(x_raw[0], target_class=target_cls)
    cf_img     = plot_counterfactual(x_raw[0], cf_result)

    # ── 6. Text summaries ─────────────────────────────────────────────────────
    top_pfi     = sorted(PFI.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    pfi_summary = "Top global drivers: " + ", ".join(
        f"{k} (Δ{v:+.4f})" for k, v in top_pfi
    )

    x_t          = torch.tensor(x_scaled, dtype=torch.float32)
    rollout      = MODEL.get_attention_rollout(x_t)
    top_attn_idx = np.argsort(rollout)[::-1][:3]
    attn_summary = "Model focused on: " + ", ".join(
        f"{FEAT_NAMES[i]} ({rollout[i]:.3f})" for i in top_attn_idx
    )

    if cf_result["found"]:
        cf_summary = (
            f"To reach '{cf_result['target']}': "
            + "; ".join(
                f"{c['feature']}: {c['original']:.1f} → {c['counterfactual']:.1f}"
                for c in cf_result["changes"][:3]
            )
        )
    else:
        if cf_result["changes"]:
            cf_summary = (
                "Best attempt (not fully converged): "
                + "; ".join(
                    f"{c['feature']}: {c['original']:.1f} → {c['counterfactual']:.1f}"
                    for c in cf_result["changes"][:3]
                )
            )
        else:
            cf_summary = "No meaningful changes found within the search budget."

    # ── 7. Response ───────────────────────────────────────────────────────────
    return jsonify({
        "prediction"    : class_names[pred],
        "pred_index"    : pred,
        "confidence"    : confidence,
        "bmi_calculated": round(float(bmi), 2),
        "xai": {
            "pfi": {
                "title"  : "Permutation Feature Importance",
                "img"    : pfi_img,
                "summary": pfi_summary,
            },
            "ale": {
                "title"  : "Accumulated Local Effects",
                "img"    : ale_img,
                "summary": (
                    "Each subplot shows how changing a feature shifts the "
                    "predicted diabetes probability. Your value is marked "
                    "with a dotted vertical line."
                ),
            },
            "attention_rollout": {
                "title"  : "Attention Rollout",
                "img"    : attn_img,
                "summary": attn_summary,
            },
            "counterfactual": {
                "title"  : "Counterfactual Explanation",
                "img"    : cf_img,
                "summary": cf_summary,
                "changes"   : cf_result["changes"],
                "cf_proba"  : cf_result["cf_proba"],
                "found"     : cf_result["found"],
                "target"    : cf_result["target"],
            },
        },
    })


@app.route("/health")
def health():
    """Liveness probe for Railway / HF Spaces health checks."""
    return jsonify({
        "status"  : "ok",
        "model"   : "ResFormer",
        "features": FEAT_NAMES,
        "ranges"  : {
            k: {"min": v[0], "max": v[1], "unit": v[3]}
            for k, v in MEDICAL_RANGES.items()
        },
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)