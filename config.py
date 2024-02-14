# data paths
SBIC_PATH = "data/SBIC/"
SBIC_TEST_PATH = f"{SBIC_PATH}SBIC.v2.tst.csv"
SBIC_TRAIN_PATH = f"{SBIC_PATH}SBIC.v2.trn.csv"
SBIC_DEV_PATH = f"{SBIC_PATH}SBIC.v2.dev.csv"
LOG_DIR = "logs/"

# model paths
GPT2_SMALL = "/bigwork/nhwpknet/models/gpt2-small"
GPT2_XL = "/bigwork/nhwpknet/models/gpt2-xl"

# random seed
SEED = 42

# model settings
MODEL_TYPE = GPT2_XL
MAX_LENGTH = 1024
WARMUP_STEPS = 500
BATCH_SIZE = 4  # from paper: 4
LEARNING_RATE = 1e-5  # from paper: 1e-5
EPOCHS = 1  # from paper: 1,2,5

# logging configuration
IC_ENABLE = True
LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = "%(asctime)s-%(levelname)s: %(message)s"
LOGGING_DATEFMT = "%d-%m-%Y %H:%M:%S"
LOGGING_FILE_DATEFMT = "%d_%m-%H_%M_%S"

# special tokens
START_TOKEN = "[STR]"
END_TOKEN = "[END]"
SEP_TOKEN = "[SEP]"
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

# sample templates
GENERATION_TEMPLATE = f"{START_TOKEN} {{post}} {SEP_TOKEN}"
TARGET_TEMPLATE = f"{{lewd}} {{off}} {{intention}} {{grp}} {SEP_TOKEN} {{group}} {SEP_TOKEN} {{statement}} {SEP_TOKEN} {{ing}} {END_TOKEN}"
