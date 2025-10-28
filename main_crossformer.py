import argparse
import os
import torch
from datetime import datetime
from model_exp.exp_train import Exp_model
from utils.tools import string_split
import random
import numpy as np

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9503))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# Setup Functions
def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='FELD-MSTM')

formatted_time = datetime.now().strftime("%m%d_%H%M")
print(formatted_time)

parser.add_argument('--model_name', type=str,  default='FELD-MSTM', help='[TESTAM, FELD-MSTM]')

parser.add_argument('--date_time', type=str,  default=formatted_time, help='date')
parser.add_argument('--data', type=str, default='TX', help='data')
parser.add_argument('--root_path', type=str, default='/home/lihaoqin/TX_datasetGen/ACTIVISg2000_meaData/case2000_2025-0101_2214/', help='lowVm-1.05 with PVandWind with traffic')
parser.add_argument('--data_path', type=str, default='CASE2000_structured_data_length12_mutiL_3to8_weak_inTransformer.npz', help='data csv file,  case14_structured_data.npz  electricity.csv')
parser.add_argument('--checkpoints', type=str, default='./log_120', help='location to store model checkpoints' ) 
parser.add_argument('--num_class', type=int, default=119, help='class_num size')
parser.add_argument('--in_len', type=int, default=12, help='input MTS length (T)')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--inchannels', type=int, default=2, help='class_num size')
parser.add_argument('--embed_size', type=int, default=64, help='192')
parser.add_argument('--num_layers', type=int, default=1, help='i')
parser.add_argument('--heads', type=int, default=2, help='i')
parser.add_argument('--out_dim', type=int, default=119, help='out dim feature')
parser.add_argument('--use_adj', type=int, default=1, help='adj feature')
parser.add_argument('--memory_size', type=int, default=30, help='memory_size')
parser.add_argument('--spatial', type=int, default=1, help='expert spatial')
parser.add_argument('--use_wpe', type=int, default=1, help='add use WaveletEncoding')
parser.add_argument('--use_FFTdcom', type=int, default=1, help='add use FFTEnhance dcomp')
parser.add_argument('--use_LD', type=int, default=1, help='add use Learnable  dcomp')
parser.add_argument('--use_LDmoe', type=int, default=0, help='add use Learnable dcomp MOE')
parser.add_argument('--use_pureSpe', type=int, default=0, help='add use use_pureSpe')
parser.add_argument('--use_ada', type=int, default=1, help='add use use adaptive meta exp')
parser.add_argument('--metaLoss', type=int, default=1, help='meta loss in middle')
parser.add_argument('--moe_gate', type=int, default=2, help='num of moe_gate')
parser.add_argument('--wpe_kernel_size', type=int, default=9, help='wpe_kernel_size')
parser.add_argument('--moving_avg', type=string_split, default='7', help='kernel size for moving average')
parser.add_argument('--heatmap', type=int, default=0, help='heatmap')
parser.add_argument('--tsne', type=int, default=0, help='tsne map')

parser.add_argument('--baseline', action='store_true', help='whether to use mean of past series as baseline for prediction', default=False)

parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=10, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=40, help='train epochs')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--seed', type=int, default=4, help='random seeds') 
parser.add_argument('--task_counter', type=int, default=0, help='task_counter')


parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
parser.add_argument('--test', type=int, default=1, help='use test')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    print(args.gpu)


set_random_seed(args.seed)
print('Args in experiment:')
print(args)

Exp = Exp_crossformer

for ii in range(args.itr):


    setting = '{}_{}_ic{}_Td{}_emb{}_layer{}_oc{}_nh{}_bs{}_seed{}'.format(args.model_name, args.data, 
                args.inchannels, args.in_len, args.embed_size, args.num_layers,
                args.out_dim, args.heads, args.batch_size, args.seed)

    exp = Exp(args) # set experiments

    if not args.test:     
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)


        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, args.save_pred)

    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, args.save_pred)