import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F



# hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  #      this can be changed to fit ur data
IMAGE_WIDTH = 256  
PIN_MEMORY = True
LOAD_MODEL = False