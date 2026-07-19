from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "pdb_raw"
FRAG_DIR = DATA_DIR / "fragments"
CKPT_DIR = ROOT / "checkpoints"
EXPORT_DIR = ROOT / "export"
CALIB_DIR = ROOT / "calibration"

# Fragment lengths we learn from PDB (matches PairFold local window)
MIN_LEN = 2
MAX_LEN = 5

# Longer sequences are segmented into [MIN_LEN, MAX_LEN] pieces
MAX_QUERY_LEN = 50000
# Max length for opening 3D in the UI (angles → client Cα-trace rebuild).
VIEW_3D_MAX_LEN = 50000
# Server-side full atom export stays capped (popup rebuilds from φ/ψ)
STRUCTURE_EXPORT_MAX_LEN = 1000
TERTIARY_MAX_LEN = 1000
# Metropolis refine only for short/medium chains; longer = score-once (fast)
TERTIARY_REFINE_MAX_LEN = 128
# Above this, skip heavy boundary clash search in SS pipeline
SS_BOUNDARY_OPT_MAX_LEN = 64
# Full SS pipeline (detect/freeze + clash_energy). clash_energy builds an N×N
# Cα distance matrix (~5 GB at 25k aa) — never run above this.
SS_PIPELINE_MAX_LEN = 256
# Clash-aware backtracking + lever correction (expensive-ish; keep bounded)
LEVER_ASSEMBLY_MAX_LEN = 256
USE_LEVER_POLISH = True  # post-SS / post-fold correct_lever_effect
# ContactPairNet anchors are only consumed by tertiary/lever (≤TERTIARY_MAX_LEN).
# Never run contacts above this — full NxN maps at 25k aa ≈ 7+ GB and freeze the OS.
CONTACT_USE_MAX_LEN = TERTIARY_MAX_LEN
# Full DP scores every 2–5 window (~4N forwards). Above this, tile greedily.
DP_FULL_MAX_LEN = 2048

# Dataset build — enhanced corpus
MAX_STRUCTURES = 1200
MAX_RESOLUTION = 2.0
FRAGMENT_STRIDE = 1
MAX_FRAGMENTS_PER_LENGTH = 120_000
# Soft dedup: keep at most this many identical sequences per length
MAX_PER_SEQUENCE = 50

# Model / train (GTX 1650 4GB)
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
UNK_IDX = len(AA_LIST)  # 20
PAD_IDX = len(AA_LIST) + 1  # 21
VOCAB_SIZE = len(AA_LIST) + 2

D_MODEL = 160
N_HEADS = 4
N_LAYERS = 4
D_FF = 320
DROPOUT = 0.12

BATCH_SIZE = 192
EPOCHS = 50
LR = 1.5e-3
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.1
SEED = 42
NUM_WORKERS = 0  # Windows-safe

# Confidence: angular error threshold for "correct" (degrees) used in calibration
CONF_ANGLE_THRESH_DEG = 25.0
# Overlap consensus weight in final confidence
CONSENSUS_WEIGHT = 0.35

# Softmax temperature sharpening (T<1 boosts high-consensus confidences)
SHARPENING_T = 0.55
DISORDER_GAMMA = 0.45

DEVICE_PREF = "cuda"

# ---------------------------------------------------------------------------
# Long-range contact head (ContactPairNet)
# ---------------------------------------------------------------------------
CONTACT_DIR = DATA_DIR / "contacts"
CONTACT_MAX_LEN = 96  # crop / pad length for training
CONTACT_INFER_MAX_LEN = 128  # full forward without cropping at inference
CONTACT_THRESH_A = 8.0  # Cα–Cα contact cutoff
CONTACT_MIN_SEP = 6  # |i−j| ≥ this for long-range
CONTACT_D_MODEL = 128
CONTACT_N_HEADS = 4
CONTACT_N_LAYERS = 3
CONTACT_D_FF = 256
CONTACT_DROPOUT = 0.12
CONTACT_BATCH_SIZE = 8
CONTACT_EPOCHS = 20
CONTACT_LR = 1e-3
CONTACT_WEIGHT_DECAY = 1e-4
CONTACT_DIST_LOSS_W = 0.25  # weight on distance regression for true contacts
CONTACT_TOP_K = 40  # max anchors passed to assembly (raised for ESM)
CONTACT_SCORE_THRESH = 0.45  # min sigmoid(logit) / ESM prob to keep
CONTACT_CROPS_PER_CHAIN = 4  # random crops written per long chain
CONTACT_MAX_CHAINS = 2000
CONTACT_CKPT_NAME = "contact_best.pt"

# ESM-2 contact backbone (Phase B). Prefer pretrained ESM contacts over
# the lightweight ContactPairNet when fair-esm is available.
USE_ESM_CONTACTS = True
ESM_MODEL_NAME = "esm2_t12_35M_UR50D"  # stable default
ESM_CONTACT_SCORE_THRESH = 0.45
ESM_CONTACT_TOP_K = 10
# Early contact-guided fold (Phase C) — before late lever/tertiary polish
EARLY_CONTACT_FOLD = True
EARLY_CONTACT_RESTARTS = 4
EARLY_CONTACT_STEPS = 320  # kept modest: each step rebuilds Cα chain
# Reject early fold if contact energy does not improve enough
EARLY_CONTACT_ACCEPT_RATIO = 0.90  # require E_after < 0.90 * E_before

# Stage-1 push: distance-geometry ensemble scaffold
USE_SCAFFOLD_ENSEMBLE = False  # MDS/mid ensemble regressed; sharp multi-seed path wins
SCAFFOLD_ENSEMBLE_MEMBERS = 3
# Optional t30 contacts — only compete in ensemble selection (never blind replace)
ESM_ALT_MODEL_NAME = "esm2_t30_150M_UR50D"
USE_ESM_ALT_CONTACTS = True  # soft-map rescue when t12 anchors gated
# Contact map quality gates
CONTACT_GATE_MIN_N = 3
CONTACT_GATE_MIN_MEAN = 0.65
CONTACT_SHARP_MEAN = 0.90  # sharper maps get multi-seed refine
CONTACT_SHARP_MIN_N = 8  # need enough anchors; 1A8O(~5) must not use heavy sharp path
# Soft full-map contact fold (torch CA optimize)
USE_SOFT_CONTACT_FOLD = True
SOFT_CONTACT_STEPS = 550
SOFT_CONTACT_THRESH = 0.10  # gated-only; keep weak signal for 1CRN

# Learned ESM fold head (Stage-1 distance refine on frozen ESM embeddings)
USE_FOLD_HEAD = False  # direct CA fold hurt RMSD; use FOLD_HEAD_SELECT instead
USE_FOLD_HEAD_SELECT = True  # rank decoys by predicted-distance MAE (safer)
FOLD_HEAD_DIR = DATA_DIR / "fold_crops"
FOLD_HEAD_CKPT_NAME = "fold_head_best.pt"
FOLD_HEAD_MAX_LEN = 96
FOLD_HEAD_MIN_LEN = 24
FOLD_HEAD_EMB_DIM = 480  # esm2_t12_35M
FOLD_HEAD_D_MODEL = 96
FOLD_HEAD_PAIR_DIM = 48
FOLD_HEAD_BATCH_SIZE = 4
FOLD_HEAD_EPOCHS = 20
FOLD_HEAD_LR = 1e-3
FOLD_HEAD_CA_STEPS = 650
FOLD_HEAD_FIT_STEPS = 1800
FOLD_HEAD_MAX_CHAINS = 800
FOLD_HEAD_CROPS_PER_CHAIN = 2

# Stage-2 / Stage-3 all-atom (backbone locked; length-capped for RAM)
USE_STAGE2_SIDECHAINS = True
USE_STAGE3_ATOMS = True
STAGE23_MAX_LEN = 256  # skip packing above this
STAGE23_ADD_HYDROGENS = False  # heavy atoms + carbonyl O by default
STAGE23_EXPORT_DIR = EXPORT_DIR / "all_atom"
