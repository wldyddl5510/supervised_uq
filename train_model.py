import argparse
import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.distributions import Normal, Independent, kl
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet, KendallProbUnet
from utils import l2_regularisation, save_mask_prediction_example
from importlib.machinery import SourceFileLoader
import matplotlib.pyplot as plt
import os
import pdb

parser = argparse.ArgumentParser(description='Supervised uncertainty quantification')
parser.add_argument('--model', type = str, default = 'vanilla_uq', help = 'vanilla or shape uq')
parser.add_argument('--output_ckpt_dir', type=str, default='/working/jnp29/checkpoints/', help='checkpoint directory')
parser.add_argument('--entcoeff', type=float, default=1.0, help='coeffient for the entropy term')
parser.add_argument('--mask1reweight', type=float, default=1.0, help='coeffient for reweighting the masks of 1s')
parser.add_argument('--trainratio', type=float, default=1.0, help='training data retention rate')
parser.add_argument('--random_seed', type=int, default=123, help='random seed')
parser.add_argument('--vb', type = bool, default = False, help = 'whether apply variational dropout to encoder')
parser.add_argument('--uq', type = bool, default = False, help = 'whether perform uq or vanilla prob Unet')
opt = parser.parse_args()
print(opt)

config_filename = 'config.py'
cf = SourceFileLoader('cf', config_filename).load_module()
small_number = torch.tensor(1e-8).cuda()

np.random.seed(opt.random_seed) # NOTE seed needs to be the same in testing due to data splitting

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dataset = LIDC_IDRI(dataset_location=cf.dataset_location)
print('dataset read complete')
dataset_size = len(dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)

split_1 = int(0.7 * dataset_size)
split_2 = int(0.85 * dataset_size)
train_indices = indices[:split_1]
val_indices = indices[split_1:split_2]
test_indices = indices[split_2:]

# Experiment with different amount of training data
print('old number of training data', len(train_indices))
train_end = int(len(train_indices) * opt.trainratio)
train_indices = train_indices[:train_end]
print('new number of training data', len(train_indices))
print('train retention rate', opt.trainratio)

train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler)

# NOTE the validation set is not used in this code
val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

# test set
test_sampler = SubsetRandomSampler(test_indices)
test_loader = DataLoader(dataset, batch_size = 1, sampler = test_sampler, shuffle = False)

print("Number of train/val/test patches:", (len(train_indices), len(val_indices), len(test_indices)))

# path
output_ckpt_filepath = opt.output_ckpt_dir + opt.model + '/vb_' + str(opt.vb) + '/uq_' + str(opt.uq) + '/net-epochs-' + str(cf.epochs) + '-ent_coeff-' + str(opt.entcoeff) + '-mask1reweight-' + str(    opt.mask1reweight) + '-train_ratio-' + str(opt.trainratio) + '-randomseed-' + str(opt.random_seed)
os.makedirs(output_ckpt_filepath, exist_ok = True)

def train(cf):
    if opt.model == 'vanilla_uq':
        net = ProbabilisticUnet(input_channels=cf.input_channels, num_classes=cf.num_classes, num_filters=cf.num_filters, latent_dim=cf.latent_dim, no_convs_fcomb=cf.no_convs_fcomb, beta=cf.beta, beta_w=cf.beta_w, vb = opt.vb)
    elif opt.model == 'shape_uq':
        net = KendallProbUnet(input_channels=cf.input_channels, num_classes=cf.num_classes, num_filters=cf.num_filters, k = cf.k, m = cf.m, no_convs_fcomb=cf.no_convs_fcomb, beta=cf.beta, beta_w=cf.beta_w)
    net.to(device)

    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=cf.adam_lr, weight_decay=cf.adam_weight_decay)
    sig = nn.Sigmoid()
    l1loss = nn.L1Loss()

    for epoch in range(cf.epochs):
        start_time = time.time()
        loss = 0.
        # print('epoch ' + str(epoch))
        for step, (patch, masks, _) in enumerate(train_loader): 
            if (step % 20) == 0:
                print('epoch ' + str(epoch) + ', step ' + str(step))
            patch = patch.to(device)
            masks = masks.to(device)
            elbo_sum = 0.
            prior_preds = []

            num_graders = masks.shape[1]

            net.unet._set_deterministic(True) # use mean of weights in unet
            unet_features = net.unet.forward(patch, False)
            net.unet._set_deterministic(False) # use random weights in unet

            for g in range(num_graders):
                mask = masks[:,g,:,:]
                mask = torch.unsqueeze(mask,1)

                # compute elbo
                net.forward(patch, mask, training=True) 
                elbo = net.elbo(mask) 
                elbo_sum += elbo

                # for entropy loss
                prior_pred = net.sample(patch, unet_features, testing=False)
                prior_preds.append(sig(prior_pred))

            prior_preds = torch.cat(prior_preds, 1) 
            mean_prior_preds = torch.mean(prior_preds, 1) 
            mean_masks = torch.mean(masks, 1) 
            
            if opt.uq:
                # entropy loss for matching aleatoric uncertainty
                ce = F.binary_cross_entropy(mean_prior_preds, mean_masks.detach(), reduction='none') 
                ce /= torch.log(torch.tensor(2.)).cuda() # log in binary_cross_entropy has base e
                entropy_loss = torch.mean(ce)

                if (step % 20) == 0:
                    print('entropy loss ' + str(entropy_loss.item()))
                
                # kl of w for variational dropout -> this is for epistemic uncertainty
                kl_w = net.unet.regularizer()
                if (step % 20) == 0:
                    print('kl_w ' + str(kl_w.item()))


            # weight regularization loss
            reg_loss = l2_regularisation(net.posterior) + \
                       l2_regularisation(net.prior) + \
                       l2_regularisation(net.fcomb.layers)


            # total loss
            inv_datalen = 1. / len(train_indices)
            # perform uq?
            if opt.uq:
                loss = -elbo_sum + opt.entcoeff * entropy_loss + cf.l2_reg_coeff * reg_loss
                # if adding variational dropout to encoder too
                # not sure: both on prior and posterior? or only on prior?
                if opt.vb:
                    kl_w_en = net.prior.regularizer()
                    if (step % 20) == 0:
                        print('kl_w_en ' + str(kl_w_en.item()))
                    loss += (inv_datalen * cf.beta_w_en * kl_w_en + inv_datalen * cf.beta_w * kl_w)
                else:
                    # all hyperparam into one w
                    loss += inv_datalen * (cf.beta_w_en + cf.beta_w) * kl_w
            # no uq -> no ent loss and kl_w loss
            else:
                loss = -elbo_sum + cf.l2_reg_coeff * reg_loss
            if (step % 20) == 0:
                print('total loss ' + str(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #print('entropy loss ' + str(entropy_loss.item()))
        #print('kl_w ' + str(kl_w.item()))
        duration = time.time() - start_time
        print('epoch ' + str(epoch) + ' took ' + str(round(duration, 2)) + ' seconds, loss: ' + str(round(loss.item(), 2)))
        if cf.save_model:
            if epoch == cf.epochs - 1:
#output_ckpt_filepath = opt.output_ckpt_dir + opt.model + '/vb_' + str(opt.vb) + '/net-epochs-' + str(epoch) + '-ent_coeff-' + str(opt.entcoeff) + '-mask1reweight-' + str(opt.mask1reweight) + '-train_ratio-' + str(opt.trainratio) + '-randomseed-' + str(opt.random_seed)
#os.makedirs(output_ckpt_filepath, exist_ok = True)
                output_ckpt_filename = output_ckpt_filepath + 'model.pt'
                # output_ckpt_filename = opt.output_ckpt_dir + opt.model + '/vb_' + str(opt.vb) + '/net-epochs-' + str(epoch) + '-ent_coeff-' + str(opt.entcoeff) + '-mask1reweight-' + str(opt.mask1reweight) + '-train_ratio-' + str(opt.trainratio) + '-randomseed-' + str(opt.random_seed) + '.pt'        
                # when saving convert to torch?
                if opt.model == 'shape_uq':
                    net.prior.encoder_mu.layers = net.prior.encoder_mu.layers.export()
                    net.posterior.encoder_mu.layers = net.posterior.encoder_mu.layers.export()
                torch.save(net.state_dict(), output_ckpt_filename)

    return net

def create_sample(cf, net):
    # TODO: Implement creating samples from trained network
    if cf.save_mask_ex:
        imgpath = output_ckpt_filepath + '/images'
        os.makedirs(imgpath, exist_ok = True)
#imgpath = output_ckpt_filepath = opt.output_ckpt_dir + opt.model + '/vb_' + str(opt.vb) + '/net-epochs-' + str(epoch) + '-ent_coeff-' + str(opt.entcoeff) + '-mask1reweight-' + str(opt.mask1reweight) + '-train_ratio-' + str(opt.trainratio) + '-randomseed-' + str(opt.random_seed) + '/images/'
        #pdb.set_trace()
        #testiter = iter(test_loader)
        for j, (patch, masks, _) in enumerate(test_loader):
            masks = torch.squeeze(masks, 0)
            patch = patch.to(device)
            # image
            plt.imshow(patch[0,:,:],cmap='Greys')
            plt.savefig(imgpath + 'base_img_' + str(j) + '.png')
            # ground truth seg
            plt.imshow(masks[0,:,:], cmap = 'Greys')
            # predicted seg
            net.eval()
            for i in range(cf.num_save_seg_per_img):
                mask_sample = net.sample(patch, None, testing = True)
                mask_sample = (torch.sigmoid(mask_sample) > 0.5).float()
                mask_sample = torch.squeeze(mask_sample, 0)
                plt.imshow(mask_sample[0,:,:], cmap = 'Greys')
                plt.savefig(imgpath + 'base_img_seg_' + str(i) + '.png')
            if j >= 10:
                break

if __name__ == "__main__":
    net = train(cf)
    create_sample(cf, net)
