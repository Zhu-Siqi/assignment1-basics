config = {
    'version': 1,
    'model': 
    {
        'vocab_size': 10000,
        'context_length': 256,
        'd_model': 512,
        'd_ff': 1344,
        'theta': 10000.0,
        'num_layers': 4,
        'num_heads': 16,
    },
    'optimizer':
    {
        'lr': 2e-4,
    },
    'scheduler':
    {
        'lr_max': 1e-3,
        'lr_min': 1e-4,
        'T_w': 500,
        'T_c': 10000,
    },
    'train':{
        'clip_grad_norm': 1.0,
        'device': 'cuda:0',
        'batch_size': 32,
        'train_steps': 12000,
        'val_interval': 500,
        'val_batches': 60,
        'val_sample_steps': 1000,
        'save_interval': 1000,
        'log_interval': 1,
    }
}

config_owt = {
    'version': 1,
    'model': 
    {
        'vocab_size': 10000,
        'context_length': 256,
        'd_model': 512,
        'd_ff': 1344,
        'theta': 10000.0,
        'num_layers': 4,
        'num_heads': 16,
    },
    'optimizer':
    {
        'lr': 2e-4,
    },
    'scheduler':
    {
        'lr_max': 1e-3,
        'lr_min': 1e-4,
        'T_w': 500,
        'T_c': 10000,
    },
    'train':{
        'clip_grad_norm': 1.0,
        'device': 'cuda:0',
        'batch_size': 32,
        'train_steps': 12000,
        'val_interval': 500,
        'val_batches': 60,
        'val_sample_steps': 1000,
        'save_interval': 1000,
        'log_interval': 1,
    }
}