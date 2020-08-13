# Default configuration value

DATA_DIR = 'flowers'
MODEL_MEAN = [0.485, 0.456, 0.406]
MODEL_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = {'train': 64, 'valid': 64, 'test': 64}
IMG_SIZE = 224
NUM_WORKERS = 0
EPOCHS = 40
DEVICE = "cuda"
LR_RATE = 0.01
NO_OF_CLASSES = 102
MODEL_NAME = 'resnet152'
SAVE_DIR = 'checkpoints'



