import argparse
import torch.backends
from exp.exp_anomaly_prediction import Exp_Anomaly_Prediction
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
    parser = argparse.ArgumentParser(description='Time Series Anomaly Prediction')

    # Basic configuration
    parser.add_argument('--task_name', type=str, default='anomaly_prediction', help='Task name')
    parser.add_argument('--is_training', type=int, default=0, help='Training mode: 1 for training, 0 for testing')
    parser.add_argument('--use_guide', type=bool, default=1, help='Whether to use guide model')
    parser.add_argument('--train_guide', type=bool, default=0, help='Whether to train guide_model')
    parser.add_argument('--cat_train', type=bool, default=1, help='Whether to use cat_train')
    parser.add_argument('--des', type=str, default='Exp_0', help='Experiment description')

    # Data loading configuration
    parser.add_argument('--data', type=str, default='MSL', help='Dataset name')
    parser.add_argument('--root_path', type=str, default='dataset/data/', help='Root path of data files')
    parser.add_argument('--data_path', type=str, default='MSL.csv', help='Data file name')
    parser.add_argument('--c_in', type=int, default=55, help='Number of input features')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location to save model checkpoints')
    parser.add_argument('--seq_len', type=int, default=32, help='Input sequence length')
    parser.add_argument('--detec_seq_len', type=int, default=64,
                        help='Detection sequence length, default is twice the prediction length')
    parser.add_argument('--pred_len', type=int, default=32, help='Output sequence length')
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--step', type=int, default=1, help='Window moving step size')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--train_epochs', type=int, default=7, help='Number of training epochs')

    # Prediction model general configuration
    parser.add_argument('--d_model', type=int, default=32, help='Prediction frequency domain loss weight')
    parser.add_argument('--d_ff', type=int, default=32, help='Prediction feedforward network dimension')
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--head_dropout', type=float, default=0.1)
    parser.add_argument('--pred_alpha', type=float, default=0.1,
                        help='Loss weight ratio between time domain and frequency domain in prediction')
    parser.add_argument('--auxi_lambda', type=float, default=0.1, help='Auxiliary loss (frequency domain loss) weight')
    parser.add_argument('--guide_lambda', type=float, default=0.1, help='Teacher guidance reconstruction loss weight')
    parser.add_argument('--dc_lambda', type=float, default=0.1, help='Dynamic contrastive loss weight')
    parser.add_argument('--detec_lambda', type=int, default=0.1,
                        help='Loss weight ratio between reconstruction and prediction')
    parser.add_argument('--score_lambda', type=float, default=0.1, help='Frequency domain score weight during testing')
    parser.add_argument('--ratio', type=int, default=list(range(0, 100)), help='Preset anomaly ratio (%)')

    # Optimization configuration
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading threads')
    parser.add_argument('--itr', type=int, default=1, help='Number of experiment repetitions')
    parser.add_argument('--lr', type=float, default=1e-4, help='Optimizer learning rate')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience value')

    # GPU configuration
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='GPU type: cuda or mps')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Whether to use multiple GPUs', default=False)
    parser.add_argument('--devices', type=str, default='0', help='Multiple GPU device IDs')

    # Model specific configuration
    # Leddam model hyperparameters
    parser.add_argument('--pe_type', type=str, default='no', help='Position embedding type')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in DEFT Block')

    # MIDformer hyperparameters
    parser.add_argument('--detec_cf_dim', type=int, default=32,
                        help='Feature dimension of frequency domain Transformer')
    parser.add_argument('--detec_d_ff', type=int, default=64, help='Feedforward network dimension')
    parser.add_argument('--detec_d_model', type=int, default=64, help='Model hidden layer dimension')
    parser.add_argument('--detec_dropout', type=float, default=0.1, help='Regular dropout rate')
    parser.add_argument('--detec_attn_dropout', type=float, default=0.1, help='Attention dropout rate')
    parser.add_argument('--detec_intra_e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--detec_inter_e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--detec_head_dim', type=int, default=32, help='Attention head dimension')
    parser.add_argument('--detec_head_dropout', type=float, default=0.1, help='Attention head dropout rate')
    parser.add_argument('--detec_n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--detec_patch_len', type=int, default=16, help='Patch size during training')
    parser.add_argument('--detec_patch_stride', type=int, default=16, help='Patch stride during training')

    # Teacher model hyperparameters
    parser.add_argument('--guide_d_model', type=int, default=64)
    parser.add_argument('--guide_d_ff', type=int, default=64)
    parser.add_argument('--guide_dropout', type=float, default=0.1)
    parser.add_argument('--guide_class_strategy', type=str, default='projection',
                        help='projection/average/cls_token')
    parser.add_argument('--guide_factor', type=int, default=1, help='Attention factor')
    parser.add_argument('--guide_e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--guide_output_attention', action='store_true',
                        help='Whether to output attention in encoder')
    parser.add_argument('--guide_use_norm', type=int, default=True, help='Use norm and denorm')
    parser.add_argument('--guide_n_heads', type=int, default=8, help='Number of heads')

    parser.add_argument('--freq', type=str, default='h',
                        help='Time feature encoding frequency: [s:second, t:minute, h:hour, d:day, b:business day, w:week, m:month], can also use finer granularity like 15min or 3h')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Time feature encoding method: [timeF, fixed, learned]')

    args = parser.parse_args()
    args = set_default_args(args)
    args.model = model
    args.guide_model = guide_model

    # Set up the equipment
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

        else:
            print(f'\n>>>>>>>>>> Only testing: {setting} <<<<<<<<<<<<\n')
            exp.test(setting, test=1)


        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    MODELS = ['Leddam_MIDformer']

    for model in MODELS:
        run_model(model)