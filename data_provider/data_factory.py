from data_provider.data_loader import DATASegLoader

from torch.utils.data import DataLoader

def data_provider(args, flag):

    data_set = DATASegLoader( args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            win_size=args.seq_len,
            step=args.step,
            flag=flag)

    shuffle_flag = False if (flag == 'test' or flag == 'thre') else True
    batch_size = args.batch_size
    drop_last = True
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
