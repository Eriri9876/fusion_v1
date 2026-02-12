import torch
import numpy as np
import random
from exp.exp_main_public import Exp_Main
import argparse
import time
import os


def main(seq_len, data_set, pred_len):
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

    seed = 1986  # 2021 2023 1995 2015 2022
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    seq_len = 30  # 30,120
    data_set = 'ETTh1'  # ['ETTh1', 'ETTm1', 'national_illness', 'exchange_rate', 'electricity']
    pred_len = 1
    args.model_name = 'PathFormer'
    args.root_path = './dataset/Public_Datasets/'
    args.label_len = 0
    args.data_name = data_set
    # args.pred_len = 720
    args.seq_len = seq_len
    args.pred_len = pred_len
    args.train_epochs = 30 // 2
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

    elif args.data_name == 'ETTh2':
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

    elif args.data_name == 'ETTm1':
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

    elif args.data_name == 'ETTm2':
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
    elif args.data_name == 'weather':
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
    elif args.data_name == 'electricity':
        args.data_path = 'electricity.csv'
        args.data = 'custom'

        args.model_id = 'electricity'
        args.model_id_name = 'electricity'
        args.data_path_name = 'electricity.csv'
        c = 321
        args.enc_in = c
        args.dec_in = c
        args.c_out = c
        args.batch_size = 16

        args.batch_size = 16
        args.residual_connection = 1
        args.num_nodes = 321
        args.layer_nums = 3
        args.k = 2
        args.d_model = 16
        args.d_ff = 128
        args.patch_size_list = [16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
    elif args.data_name == 'exchange_rate':
        args.data = 'custom'
        args.data = 'exchange_rate'
        args.data_path = 'exchange_rate.csv'

        args.model_id_name = 'exchange_rate'
        args.data_path_name = 'exchange_rate.csv'

        c = 8
        args.enc_in = c
        args.dec_in = c
        args.c_out = c

        args.num_nodes = c
        args.layer_nums = 3
        args.k = 2
        args.d_model = 16
        args.d_ff = 64

        args.patch_size_list = [16, 12, 8, 32, 12, 8, 6, 32, 8, 6, 16, 12]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
    elif args.data_name == 'national_illness':
        args.data_path = 'national_illness.csv'
        args.data = 'custom'
        args.data = 'national_illness'
        args.model_id_name = 'national_illness'
        args.data_path_name = 'national_illness.csv'
        c = 7
        args.enc_in = c
        args.dec_in = c
        args.c_out = c
        args.batch_size = 16
        args.num_nodes = c
        args.layer_nums = 3
        args.k = 2
        args.d_model = 16
        args.d_ff = 64

        args.patch_size_list = [16, 12, 8, 32, 12, 8, 6, 32, 8, 6, 16, 12]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()

    if seq_len == 30:
        args.layer_nums = 2
        args.patch_size_list = [16, 12, 8, 12, 8, 6, 8, 6, 16, 12]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
    args.learning_rate = 1e-3  # or 1e-4 select the best result
    args.checkpoints = './checkpoints_pathformer_shortterm/'
    args.explore_fund_memory = False
    Exp = Exp_Main
    print(args.batch_size)
    args.batch_size = 128
    if args.data_name == 'electricity' or args.data_name == 'national_illness':
        args.batch_size = 16
    if args.data_name == 'electricity':
        args.d_model = args.d_model
    else:
        args.d_model = args.d_model * 2

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
            #
            # time_now = time.time()
            # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test(setting)
            # print('Inference time: ', time.time() - time_now)
            #
            # if args.do_predict:
            #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #     exp.predict(setting, True)

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
    seq_len_all = [96, 96 * 4, 96 * 6]
    # seq_len_all=[30]
    seq_len_all = [30, 120]
    data_set_all = ['ETTh1', 'ETTm1', 'weather', 'ETTh2', 'ETTm2']
    data_set_all = ['ETTh1', 'ETTm1', 'national_illness', 'exchange_rate', 'electricity']
    data_set_all = ['electricity']
    pre_len_all = [96, 192, 336, 720]
    # seq_len_all=[30]
    pre_len_all = [1]
    for seq_len in seq_len_all:
        for data_set in data_set_all:
            for pre_len in pre_len_all:
                main(seq_len, data_set, pre_len)
