import torch
import numpy as np
import random
from exp.exp_main_public import Exp_Main
import argparse
import time
import os

fix_seed = 1024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def main():
    parser = argparse.ArgumentParser(description='Multivariate Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, default='PathFormer',
                        help='model name, options: [PathFormer]')
    parser.add_argument('--model_id', type=str, default="ETT.sh")

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/weather', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S]; M:multivariate predict multivariate, S:univariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # model
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--num_nodes', type=int, default=21)
    parser.add_argument('--layer_nums', type=int, default=3)
    parser.add_argument('--k', type=int, default=2, help='choose the Top K patch size at the every layer ')
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int,
                        default=[16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2])  # 16 12 8 32 12 8 6 4 8 6 4 2
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')
    parser.add_argument('--drop', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--metric', type=str, default='mae')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='2', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    seq_len = 96 * 4  # 96 96*6
    data_set = 'weather'  # ['ETTh1','ETTm1','weather','ETTh2','ETTm2']
    pred_len = 720  # [96,192,336,720]

    args.model_name = 'PathFormer'
    args.root_path = './dataset/Public_Datasets/'
    args.label_len = 0

    args.data_name = data_set
    args.seq_len = seq_len
    args.pred_len = pred_len
    args.gpu = 5
    args.itr = 1
    args.speed_mode = True
    args.patch_squeeze = False
    args.record = True

    if args.data_name == 'ETTh1':
        args.data = args.data_name
        args.data_path_name = 'ETTh1.csv'
        args.model_id_name = 'ETTh1'
        args.data_path = args.data_path_name
        args.batch_size = 128
        args.residual_connection = 1
        args.num_nodes = 7
        args.layer_nums = 3
        args.k = 3
        args.d_model = 4
        args.d_ff = 64
        args.patch_size_list = [16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
    if args.data_name == 'ETTh2':
        args.data = args.data_name
        args.data_path_name = 'ETTh2.csv'
        args.model_id_name = 'ETTh2'
        args.data_path = args.data_path_name
        args.batch_size = 128

        args.residual_connection = 0
        args.num_nodes = 7
        args.layer_nums = 3
        args.k = 2
        args.d_model = 4
        args.d_ff = 64
        args.patch_size_list = [16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()

    if args.data_name == 'ETTm1':
        args.data = args.data_name
        args.data_path_name = 'ETTm1.csv'
        args.model_id_name = 'ETTm1'
        args.data_path = args.data_path_name
        args.batch_size = 128

        args.num_nodes = 7
        args.layer_nums = 3
        args.k = 3
        args.d_model = 8
        args.d_ff = 64
        args.patch_size_list = [16, 12, 8, 4, 12, 8, 6, 4, 8, 6, 2, 12]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()

    if args.data_name == 'ETTm2':
        args.data = args.data_name
        args.data_path_name = 'ETTm2.csv'
        args.model_id_name = 'ETTm2'
        args.data_path = args.data_path_name
        args.batch_size = 128

        args.num_nodes = 7
        args.layer_nums = 3
        args.k = 2
        args.d_model = 16
        args.d_ff = 64
        args.patch_size_list = [16, 12, 8, 32, 12, 8, 6, 32, 8, 6, 16, 12]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
    if args.data_name == 'weather':
        args.data = 'custom'
        args.data_path_name = 'weather.csv'
        args.model_id_name = 'weather'
        args.data_path = args.data_path_name
        args.batch_size = 128 // 2  # 256

        args.num_nodes = 21
        args.layer_nums = 3
        args.k = 2
        args.d_model = 8
        args.d_ff = 64
        args.patch_size_list = [16, 12, 8, 4, 12, 8, 6, 4, 8, 6, 2, 12]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
    args.checkpoints = './checkpoints_pathformer_longterm/'
    args.script_id = '0_'
    Exp = Exp_Main

    args.batch_size = 128
    args.learning_rate = 1e-4  # or 1e-3 select the best result
    args.explore_fund_memory = False
    if args.explore_fund_memory:
        args.enc_in = 2
        args.num_nodes = 2
    args.is_training = True
    print('Args in experiment:')
    print(args)
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_ft{}_sl{}_pl{}_{}'.format(
                args.model,
                args.data_path[:-4],
                args.seq_len,
                args.pred_len, ii)
            args.save_path = os.path.join(args.checkpoints, setting)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            exp = Exp(args)  # set experiments

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            time_now = time.time()
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            print('Inference time: ', time.time() - time_now)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_ft{}_sl{}_pl{}_{}'.format(
            args.model_id,
            args.model,
            args.data_path[:-4],
            args.features,
            args.seq_len,
            args.pred_len, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
