from torch.utils.data import DataLoader
from .dataset_plax_kf import LMData_US

def get_dataloader(config):
    data_dir = config.data_dir
    batch_size = config.batch_size

    train_dataset = LMData_US()
    test_dataset = LMData_US(mode='val')

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              num_workers=config.num_work, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                             num_workers=config.num_work, shuffle=False)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))

    data_loader = {'train': train_loader, 'test': test_loader}

    return data_loader
