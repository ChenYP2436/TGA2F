import argparse
import torch.backends
from exp.exp_AAAA_cat_loss2_0915 import Exp_Anomaly_Prediction
import random
import numpy as np

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

DEFAULT_MASK_HYPER_PARAMS = {
    "Mlr": 1e-5,
    "individual": 0,
    "auxi_loss": "MAE",
    "auxi_type": "complex",
    "auxi_mode": "fft",
    "regular_lambda": 0.5,
    "inference_patch_stride": 1,
    "inference_patch_size": 16,
    "module_first": True,
    "mask": False,
    "pretrained_model": None,
    "pct_start": 0.3,
    "revin": 1,
    "detec_affine": 0,
    "detec_subtract_last": 0,
    "detec_temperature": 0.07,
    "lradj": "type1",
}


def set_default_args(args):
    defaults = DEFAULT_MASK_HYPER_PARAMS
    if hasattr(args, '__dict__'):
        for key, value in defaults.items():
            if not hasattr(args, key):
                setattr(args, key, value)
    else:
        for key, value in defaults.items():
            if key not in args:
                args[key] = value
    return args

guide_model = 'iTransGuide'

def run_model(model):
    parser = argparse.ArgumentParser(description='时间序列异常预测')

    # 基础配置
    parser.add_argument('--task_name', type=str, default='anomaly_prediction', help='任务名称')
    parser.add_argument('--is_training', type=int, default=0, help='训练模式: 1表示训练, 0表示测试')
    parser.add_argument('--use_guide', type=bool, default=1, help='是否使用guide model')
    parser.add_argument('--train_guide', type=bool, default=0, help='是否需要训练guide_model')
    parser.add_argument('--cat_train', type=bool, default=1, help='是否使用cat_train')
    parser.add_argument('--des', type=str, default='Exp_test', help='实验描述')


    # 数据加载配置
    parser.add_argument('--data', type=str, default='ASD_dataset_1', help='数据集名称')
    parser.add_argument('--root_path', type=str, default='dataset/data/', help='数据文件根目录')
    parser.add_argument('--data_path', type=str, default='ASD_dataset_1.csv', help='数据文件名')
    parser.add_argument('--c_in', type=int, default=19, help='输入特征数')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_AAA20920/', help='模型检查点保存位置')
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--detec_seq_len', type=int, default=192, help='检测序列长度,默认为预测长度的两倍')
    parser.add_argument('--pred_len', type=int, default=32, help='输出序列长度')
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--step', type=int, default=1, help='每次窗口移动步长')
    parser.add_argument('--batch_size', type=int, default=64, help='训练批次大小')
    parser.add_argument('--train_epochs', type=int, default=7, help='训练轮数')

    # 预测模型通用配置
    parser.add_argument('--d_model', type=int, default=32, help='预测频域损失权重')
    parser.add_argument('--d_ff', type=int, default=32, help='预测前馈网络维度')
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--head_dropout', type=float, default=0.1)
    parser.add_argument('--pred_alpha', type=float, default=0.1, help='预测中时域和频域的损失权重比例')
    parser.add_argument('--auxi_lambda', type=float, default=0.0, help='辅助损失(频域损失)权重')
    parser.add_argument('--guide_lambda', type=float, default=0.01, help='教师指导重构损失权重')
    parser.add_argument('--dc_lambda', type=float, default=0.5, help='动态对比损失权重')
    parser.add_argument('--detec_lambda', type=int, default=0., help='重构和预测的损失权重比例')
    parser.add_argument('--score_lambda', type=float, default=0., help='测试时频域分数权重')
    parser.add_argument('--ratio', type=int, default=list(range(0, 100)), help='预设异常比例(%)')

    # 优化配置
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--itr', type=int, default=1, help='实验重复次数')
    parser.add_argument('--lr', type=float, default=1e-4, help='优化器学习率')
    parser.add_argument('--patience', type=int, default=3, help='早停耐心值')

    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='GPU类型: cuda或mps')
    parser.add_argument('--use_multi_gpu', action='store_true', help='是否使用多GPU', default=False)
    parser.add_argument('--devices', type=str, default='0', help='多GPU设备ID')

    # 模型特定配置
    # Leddam 模型超参
    parser.add_argument('--pe_type', type=str, default='no', help='position embedding type')
    parser.add_argument('--n_layers', type=int, default=3, help='n_layers of DEFT Block')



    # CATCH 超参
    parser.add_argument('--detec_cf_dim', type=int, default=32, help='频域Transformer的特征维度')
    parser.add_argument('--detec_d_ff', type=int, default=64, help='前馈网络维度')
    parser.add_argument('--detec_d_model', type=int, default=64, help='模型隐藏层维度')
    parser.add_argument('--detec_dropout', type=float, default=0.1, help='普通dropout率')
    parser.add_argument('--detec_attn_dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--detec_intra_e_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--detec_inter_e_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--detec_head_dim', type=int, default=32, help='注意力头维度')
    parser.add_argument('--detec_head_dropout', type=float, default=0.1, help='注意力头dropout率')
    parser.add_argument('--detec_n_heads', type=int, default=4, help='注意力头数量')
    parser.add_argument('--detec_patch_len', type=int, default=16, help='训练时patch大小')
    parser.add_argument('--detec_patch_stride', type=int, default=16, help='训练时patch步长')

    if 'iTransGuide' in guide_model:
        parser.add_argument('--guide_d_model', type=int, default=64)
        parser.add_argument('--guide_d_ff', type=int, default=64)
        parser.add_argument('--guide_dropout', type=float, default=0.1)
        parser.add_argument('--guide_class_strategy', type=str, default='projection',
                            help='projection/average/cls_token')
        parser.add_argument('--guide_factor', type=int, default=1, help='attn factor')
        parser.add_argument('--guide_e_layers', type=int, default=2, help='编码器层数')
        parser.add_argument('--guide_output_attention', action='store_true',
                            help='whether to output attention in ecoder')
        parser.add_argument('--guide_use_norm', type=int, default=True, help='use norm and denorm')
        parser.add_argument('--guide_n_heads', type=int, default=8, help='num of heads')



    # 不重要超参
    parser.add_argument('--freq', type=str, default='h', help='时间特征编码频率: [s:秒, t:分钟, h:小时, d:天, b:工作日, w:周, m:月], 也可用更细粒度如15min或3h')
    parser.add_argument('--embed', type=str, default='timeF', help='时间特征编码方式: [timeF, fixed, learned]')

    args = parser.parse_args()
    args = set_default_args(args)
    args.model = model
    args.guide_model = guide_model

    # 设置设备
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print(f'Using GPU for {model}')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print(f'Using cpu or mps for {model}')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    Exp = Exp_Anomaly_Prediction
    for ii in range(args.itr):
        exp = Exp(args)
        setting = 'bs{}_len{}_step{}_dm{}_df{}_{}'.format(
            args.batch_size, args.seq_len, args.step,
            args.d_model, args.d_ff, args.des)

        if args.is_training == 1:
            print(f'\n>>>>>>>>>> Start training: {setting} >>>>>>>>>>>>\n')
            exp.train(setting)

            print(f'\n>>>>>>>>>> Start testing: {setting} <<<<<<<<<<<<\n')
            exp.test(setting)

        elif args.is_training == 0:
            print(f'\n>>>>>>>>>> Only testing: {setting} <<<<<<<<<<<<\n')
            exp.test(setting, test=1)

        else:
            print(f'\n>>>>>>>>>> Showing result: {setting} <<<<<<<<<<<<\n')
            exp.show_result(setting)

        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    MODELS = ['Leddam_AAA2']

    for model in MODELS:
        run_model(model)