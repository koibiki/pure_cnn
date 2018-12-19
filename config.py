from easydict import EasyDict as edict
import os
import os.path as osp

cfg = edict()

cfg.PATH = edict()
cfg.PATH.ROOT_DIR = os.getcwd()
cfg.PATH.TBOARD_SAVE_DIR = osp.abspath(osp.join(os.getcwd(), 'logs'))
cfg.PATH.MODEL_SAVE_DIR = osp.abspath(osp.join(os.getcwd(), 'checkpoints'))
cfg.PATH.TFLITE_MODEL_SAVE_DIR = osp.abspath(osp.join(os.getcwd(), 'tf_lite_model'))

cfg.TRAIN = edict()
# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.LEARNING_RATE = 0.005
cfg.TRAIN.LR_DECAY_STEPS = 10000
cfg.TRAIN.LR_DECAY_RATE = 0.9
cfg.TRAIN.EPOCHS = 50000
cfg.TRAIN.DISPLAY_STEP = 100
cfg.TRAIN.GPU_MEMORY_FRACTION = 0.5
cfg.TRAIN.TF_ALLOW_GROWTH = True

# TEST
cfg.TEST = edict()
cfg.TEST.BATCH_SIZE = 1
