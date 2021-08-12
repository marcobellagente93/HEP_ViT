######################
# Choose dataset: #
######################

dataset = ''

##############
# Training:  #
##############

lr = 5e-3
batch_size = 512
gamma = 0.95
weight_decay = 0.
betas = (0.9, 0.999)

n_epochs = 200
n_its_per_epoch = 1

#################
# Architecture: #
#################

img_size = 40
patch_size = 20

in_dim = 32
depth = 2
heads = 8
mlp_dim = 32

num_classes = 2
channels = 1

####################
# Logging/preview: #
####################

progress_bar = True                         # Show a progress bar of each epoch

show_interval = 50
save_interval = 10

###################
# Loading/saving: #
###################

test = False
train = True

save_model = True
load_model = False

save_dir = './experiments'
checkpoint_on_error = False
