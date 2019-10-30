######### global settings  #########
GPU = True                                  # running on GPU is highly suggested
TEST_MODE = False                           # turning on the testmode means the code will run on a small dataset.
CLEAN = True                               # set to "True" if you want to clean the temporary large files after generating result
MODEL = 'toy_data_model'                          # model arch: resnet18, alexnet, resnet50, densenet161, toy_data_model
DATASET = 'toy_data'                       # model trained on: places365 or imagenet or places50 or toy_data
QUANTILE = 0.005                            # the threshold used for activation
SEG_THRESHOLD = 0.04                        # the threshold used for visualization
SCORE_THRESHOLD = 0.08                      # the threshold used for IoU score (in HTML file)
TOPN = 10                                   # to show top N image with highest activation for each unit
PARALLEL = 1                                # how many process is used for tallying (Experiments show that 1 is the fastest)
#CATAGORIES = ["object", "part","scene","texture","color"] # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
CATAGORIES = ["color","shape","shape_color"]
OUTPUT_FOLDER = "/home/mindgarage05/magus/projects/netdissect-lite/result/pytorch_"+\
                MODEL+"_thresh"+str(SCORE_THRESHOLD)+"_"+DATASET # result will be stored in this folder

########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

if MODEL == 'alexnet':
    DATA_DIRECTORY = '/scratch/data/broden1_227'
    IMG_SIZE = 224
elif MODEL == 'toy_data_model':
    DATA_DIRECTORY = '/scratch/data/synthetic_labels_v2'
    IMG_SIZE = 64

else:
    DATA_DIRECTORY = '/scratch/data/broden1_224'
    IMG_SIZE = 227



if DATASET == 'places365':
    NUM_CLASSES = 365
elif DATASET == 'imagenet':
    NUM_CLASSES = 1000
elif DATASET == 'places50' or DATASET == 'places50_reduced':
    NUM_CLASSES = 50
elif DATASET == 'toy_data':
    NUM_CLASSES = 2
if MODEL == 'resnet18':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/resnet18_places365.pth.tar'
        MODEL_PARALLEL = True
    elif DATASET == 'imagenet':
        MODEL_FILE = None
        MODEL_PARALLEL = False
elif MODEL == 'densenet161':
    FEATURE_NAMES = ['features']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/whole_densenet161_places365_python36.pth.tar'
        MODEL_PARALLEL = False
elif MODEL == 'resnet50':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        #MODEL_FILE = 'zoo/whole_resnet50_places365_python36.pth.tar'
        MODEL_FILE = 'zoo/resnet50_places365.pth.tar'
        MODEL_PARALLEL = True

elif MODEL == 'alexnet':
    FEATURE_NAMES = ['conv3','conv4', 'conv5']
    if DATASET == 'places365' or DATASET == 'places50':
        FOLDER_NAME= 'l_2_1_lr0.01_wd0_pen0.005_act0_spa0.01_b256_c50_id49'
        MODEL_FILE = '/home/mindgarage05/magus/back_up_thesis/saved_models/mg03_models_runs/' \
                     'zoo/'+FOLDER_NAME+'/alexnet_best.pth.tar'
        MODEL_PARALLEL = True
        OUTPUT_FOLDER = OUTPUT_FOLDER+"_"+FOLDER_NAME

elif MODEL == 'toy_data_model':
    FEATURE_NAMES = ['conv1']
    if DATASET == 'toy_data':
        FOLDER_NAME= 'custom2_l_2_1_lr0.01_wd0_pen0.01_act0.001_spa0_b32_c2_orth0.001_id161'
        MODEL_FILE = '/home/mindgarage05/magus/projects/pycharm_projec46/zoo/'+FOLDER_NAME+'/custom_model_1_best.pth.tar'
        MODEL_PARALLEL = True
        OUTPUT_FOLDER = OUTPUT_FOLDER+"_"+FOLDER_NAME


if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    INDEX_FILE = 'index_sm.csv'
    OUTPUT_FOLDER += "_test"
else:
    WORKERS = 12
    BATCH_SIZE = 128  # for resent50: 64
    TALLY_BATCH_SIZE = 16
    TALLY_AHEAD = 4
    INDEX_FILE = 'index.csv'
