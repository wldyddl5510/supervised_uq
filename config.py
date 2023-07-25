# The training and testing scripts share the same config file 

import os
import numpy as np
from collections import OrderedDict

config_path = os.path.realpath(__file__)

model = "shape_uq"
input_channels = 1
num_classes = 1
num_filters = [32,64,128,192]
latent_dim = 6
k = 4
m = 2
no_convs_fcomb = 4
beta = 1.0 # for kl[q(z|x,y) || p(z|x)]
beta_w = 0.5 # for kl[q(w) || p(w)]
beta_w_en = 0.5
adam_lr = 1e-6
adam_weight_decay = 0
epochs = 1
l2_reg_coeff = 1e-5
train_retention_rate = 1.
dataset_location = '/users/jnp29/gunay/Probabilistic-Neural-Networks/LIDC_data/'
save_model = False
save_mask_ex = True
num_save_base_img = 10
num_save_seg_per_img = 10
