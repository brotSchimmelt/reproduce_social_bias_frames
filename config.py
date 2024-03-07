# logging configuration
LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = "%(asctime)s-%(levelname)s: %(message)s"
LOGGING_DATEFMT = "%d-%m-%Y %H:%M:%S"
LOGGING_FILE_DATEFMT = "%d_%m-%H_%M_%S"

# paths
DATA_PATH = "data/"
SBIC_PATH = f"{DATA_PATH}SBIC/"
SBIC_TEST_PATH = f"{SBIC_PATH}SBIC.v2.tst.csv"
SBIC_TRAIN_PATH = f"{SBIC_PATH}SBIC.v2.trn.csv"
SBIC_DEV_PATH = f"{SBIC_PATH}SBIC.v2.dev.csv"
DEV_EVAL_PROMPTS = f"{DATA_PATH}dev_eval_prompts.csv"
TEST_EVAL_PROMPTS = f"{DATA_PATH}test_eval_prompts.csv"
MODEL_PATH = "tmp/models/"
OUTPUT_PATH = "tmp/output/"
LOG_DIR = "logs/"
CHECKPOINT_DIR = "tmp/checkpoints/"

# model paths
GPT2_SMALL = "/bigwork/nhwpknet/models/gpt2-small"
GPT2_XL = "/bigwork/nhwpknet/models/gpt2-xl"
DEFAULT_MODEL = GPT2_XL  # version of GPT2 not mentioned in paper

# random seeds
DEFAULT_SEED = 42
EXPERIMENT_SEEDS = [42, 1337, 31415, 271828, 1701]

# model settings
MAX_LENGTH = 128  # not mentioned in paper
LOGGING_STEPS = 500
DEFAULT_WARMUP_STEPS = 500  # paper did mention warum up, but not the number of steps
DEFAULT_BATCH_SIZE = 4  # from paper: 4
DEFAULT_LEARNING_RATE = 1e-5  # from paper: 1e-5
DEFAULT_NUM_EPOCHS = 5  # from paper: 1,2,5

# inference settings
INFERENCE_BATCH_SIZE_GREEDY = 256  # max for 256 token model
INFERENCE_BATCH_SIZE_SAMPLING = 32  # max for 256 token model
PADDING_SIDE = "left"
NUM_RETURN_SEQ = 10  # from paper for sampling based inference: 10

# special tokens
START_TOKEN = "<|startoftext|>"
END_TOKEN = "<|endoftext|>"
SEP_TOKEN = "<|sep|>"
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
HELP_SEP = "[SEP]"
OTHER_TOKENS = [
    "[offN]",
    "[offY]",
    "[lewdN]",
    "[lewdY]",
    "[intN]",
    "[intY]",
    "[grpN]",
    "[grpY]",
    "[ingN]",
    "[ingY]",
    "[SEP]",
]
OFF_TOKEN = {0: "[offN]", 1: "[offY]"}
LEWD_TOKEN = {
    0: "[lewdN]",
    1: "[lewdY]",
}
INT_TOKEN = {
    0: "[intN]",
    1: "[intY]",
}
GRP_TOKEN = {
    0: "[grpN]",
    1: "[grpY]",
}
ING_TOKEN = {
    0: "[ingN]",
    1: "[ingY]",
}

# template for generation
GENERATION_TEMPLATE = f"{START_TOKEN} {{post}} {SEP_TOKEN}"

# templates for training
TRAIN_TEMPLATE_FULL = f"{START_TOKEN} {{post}} {SEP_TOKEN} {{lewd}} {{off}} {{intention}} {{grp}} {SEP_TOKEN} {HELP_SEP} {{group}} {SEP_TOKEN} {HELP_SEP} {{statement}} {SEP_TOKEN} {HELP_SEP} {{ing}} {END_TOKEN}"
TRAIN_TEMPLATE_OFFN = f"{START_TOKEN} {{post}} {SEP_TOKEN} {{lewd}} {{off}} {END_TOKEN}"
TRAIN_TEMPLATE_GRPN = f"{START_TOKEN} {{post}} {SEP_TOKEN} {{lewd}} {{off}} {{intention}} {{grp}} {END_TOKEN}"
