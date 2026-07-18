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
CONTACT_TOP_K = 20  # max anchors passed to assembly
CONTACT_SCORE_THRESH = 0.45  # min sigmoid(logit) to keep
CONTACT_CROPS_PER_CHAIN = 4  # random crops written per long chain
CONTACT_MAX_CHAINS = 2000
CONTACT_CKPT_NAME = "contact_best.pt"
