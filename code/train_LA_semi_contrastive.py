import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import CosineSimilarity
from torch.utils.data import DataLoader

from networks.E2DNet import VNet_Encoder, MainDecoder, TriupDecoder, center_model

from utils import losses
from dataloaders.la_heart import LAHeart, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='data path')
parser.add_argument('--model', type=str,  default='SCC', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='number of labeled data')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')
parser.add_argument('--has_triup', type=int,  default=True, help='whether adopted triup decoder as auxiliary decoder')

# loss
parser.add_argument('--my_lambda', type=float,  default=1, help='balance factor to control contrastive loss')
parser.add_argument('--tau', type=float,  default=1, help='temperature of the contrastive loss')

parser.add_argument('--has_contrastive', type=int,  default=1, help='whether use contrative loss')
parser.add_argument('--only_supervised', type=int,  default=0, help='whether use consist loss')

args = parser.parse_args()
exp_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

train_data_path = args.root_path
snapshot_path = "../semi_model/" + args.model + "/" + exp_time + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True #
    cudnn.deterministic = False #
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Network definition
    # E2DNet for segmentation
    encoder = VNet_Encoder(n_channels=1, n_filters=16, normalization='batchnorm',has_dropout=True).cuda()
    seg_decoder_1 = MainDecoder(n_classes=num_classes, n_filters=16, normalization='batchnorm',has_dropout=True).cuda()
    if args.has_triup:
        # adopted this decoder, the performance will hit dice=0.9056, 59hd=6.74
        seg_decoder_2 = TriupDecoder(n_classes=num_classes, n_filters=16, normalization='batchnorm',has_dropout=True).cuda()
    else:
        seg_decoder_2 = MainDecoder(n_classes=num_classes, n_filters=16, normalization='batchnorm',has_dropout=True).cuda()
    seg_params = list(encoder.parameters())+list(seg_decoder_1.parameters())+list(seg_decoder_2.parameters())
    # classification model
    center_model = center_model(num_classes=num_classes,ndf=64)
    center_model.cuda()
    db_train = LAHeart(base_dir=train_data_path,
                       split='train80', # train/val split
                       # num=args.labelnum,#16,
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))

    labelnum = args.labelnum    # default 16
    label_idx = list(range(0,80))
    random.shuffle(label_idx)
    labeled_idxs = label_idx[:labelnum]
    unlabeled_idxs = label_idx[labelnum:80]
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(seg_params, lr=base_lr, momentum=0.9, weight_decay=0.0001)
    cos_sim = CosineSimilarity(dim=1,eps=1e-6)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            features = encoder(volume_batch)
            outputs_1 = seg_decoder_1(features)
            outputs_2 = seg_decoder_2(features)
            outputs_soft_1 = F.softmax(outputs_1,dim=1)
            outputs_soft_2 = F.softmax(outputs_2,dim=1)
            
            '''
            # fully supervised manner
            loss_seg_1 = F.cross_entropy(outputs_1, label_batch)#ce_loss(outputs_1[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())
            loss_seg_dice_1 = losses.dice_loss(outputs_soft_1[:, 1, :, :, :], label_batch == 1)
            loss_seg_2 = F.cross_entropy(outputs_2, label_batch)
            loss_seg_dice_2 = losses.dice_loss(outputs_soft_2[:,1, :, :, :], label_batch == 1)
            '''
            ## calculate the supervised loss
            loss_seg_1 = F.cross_entropy(outputs_1[:labeled_bs, ...], label_batch[:labeled_bs])#ce_loss(outputs_1[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())
            loss_seg_dice_1 = losses.dice_loss(outputs_soft_1[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            loss_seg_2 = F.cross_entropy(outputs_2[:labeled_bs, ...], label_batch[:labeled_bs])
            loss_seg_dice_2 = losses.dice_loss(outputs_soft_2[:labeled_bs,1, :, :, :], label_batch[:labeled_bs] == 1)
            
            supervised_loss = 0.5*(loss_seg_dice_1 + loss_seg_dice_2) + 0.5*(loss_seg_1 + loss_seg_2)
            
            if args.has_contrastive == 1:
                
                create_center_1_bg = center_model(outputs_1[:,0,...].unsqueeze(1))# 4,1,x,y,z->4,2
                create_center_1_la = center_model(outputs_1[:,1,...].unsqueeze(1))
                create_center_2_bg = center_model(outputs_2[:,0,...].unsqueeze(1))
                create_center_2_la = center_model(outputs_2[:,1,...].unsqueeze(1))
        
                create_center_soft_1_bg = F.softmax(create_center_1_bg, dim=1)# dims(4,2)
                create_center_soft_1_la = F.softmax(create_center_1_la, dim=1)
                create_center_soft_2_bg = F.softmax(create_center_2_bg, dim=1)# dims(4,2)
                create_center_soft_2_la = F.softmax(create_center_2_la, dim=1)
                
                lb_center_12_bg = torch.cat((create_center_soft_1_bg[:labeled_bs,...], create_center_soft_2_bg[:labeled_bs,...]),dim=0)# 4,2
                lb_center_12_la = torch.cat((create_center_soft_1_la[:labeled_bs,...], create_center_soft_2_la[:labeled_bs,...]),dim=0)
                un_center_12_bg = torch.cat((create_center_soft_1_bg[labeled_bs:,...], create_center_soft_2_bg[labeled_bs:,...]),dim=0)
                un_center_12_la = torch.cat((create_center_soft_1_la[labeled_bs:,...], create_center_soft_2_la[labeled_bs:,...]),dim=0)
               

                # cosine similarity
                loss_contrast = losses.scc_loss(cos_sim, args.tau, lb_center_12_bg,lb_center_12_la, un_center_12_bg, un_center_12_la)
                loss = supervised_loss + args.my_lambda * loss_contrast
                
            if args.only_supervised==1:
                print('only supervised')
                loss = supervised_loss
            
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg_1+loss_seg_2, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice_1+loss_seg_dice_2, iter_num)
            writer.add_scalar('loss/loss_supervised', supervised_loss, iter_num)
            if args.has_contrastive == 1:
                writer.add_scalar('loss/loss_contrastive', loss_contrast, iter_num)
            


            logging.info(
                'iteration %d : loss : %f, loss_seg_1: %f, loss_seg_2: %f, loss_dice_1: %f, loss_dice_2: %f' %
                (iter_num, loss.item(),   loss_seg_1.item(), loss_seg_2.item(), loss_seg_dice_1.item(), loss_seg_dice_2.item()))
            if args.has_contrastive == 1:
                logging.info(
                'iteration %d : supervised loss : %f, contrastive loss: %f' %
                (iter_num, supervised_loss.item(),  loss_contrast.item()))

            ## change lr
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            ## save checkpoint
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save({'encoder_state_dict':encoder.state_dict(),
                            'seg_decoder_1_state_dict': seg_decoder_1.state_dict(),
                            'seg_decoder_2_state_dict': seg_decoder_2.state_dict()}, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
