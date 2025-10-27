from cs336_basics.model import CustomTransformerLM
from cs336_basics.config import config
from cs336_basics.train_modules import *
import swanlab
import numpy as np
from tqdm import tqdm

swanlab.login(api_key="BdUCYhXh01FutuwBb4uWk", save=True)

TRAIN_PATH = 'data/TinyStoriesV2-GPT4-train.npy'
VALID_PATH = 'data/TinyStoriesV2-GPT4-valid.npy'
SAVE_PATH = './check_point'
CHECKPOINT_PATH = './check_point/checkpoint.pt'

def main(config = config):
    meta_config = config['model'].copy()
    meta_config.update(config['optimizer'])
    meta_config.update(config['train'])

    swanlab.init(
        project = f'demo_LM_{config['version']}',
        
        config= meta_config
        
    )
    
    device = torch.device(config['train']['device'])
    criterion = CustomCrossEntropyLoss()

    model = CustomTransformerLM(**config['model']).to(device)
    optimizer = CustomAdamW(model.parameters(), **config['optimizer'])
    if os.path.isfile(CHECKPOINT_PATH):
        start_iteration = load_checkpoint(CHECKPOINT_PATH, model, optimizer)
    else:
        print("No checkpoint found. Starting training from scratch.")
        start_iteration = 0
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

    train_data = np.load(TRAIN_PATH)
    valid_data = np.load(VALID_PATH)

    for iterations in tqdm(range(start_iteration, config['train']['train_steps'])):
        # train
        model.train()
        train_x, train_y = get_batch(train_data, config['train']['batch_size'], 
                                     config['model']['context_length'], device)

        out_logits = model(train_x)
        loss = criterion(out_logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), config['train']['clip_grad_norm'])

        lr_t = get_lr(iterations, **config['scheduler'])
        for group in optimizer.param_groups:
            group['alpha'] = lr_t
        optimizer.step()

        # valid
        if iterations % config['train']['val_interval'] == 0:
            model.eval()
            with torch.no_grad():
                val_loss_list = []
                for _ in range(config['train']['val_sample_steps']):
                    valid_x, valid_y = get_batch(valid_data, config['train']['val_batches'], 
                                                 config['model']['context_length'], device)
                    val_loss = criterion(model(valid_x), valid_y)
                    val_loss_list.append(val_loss.cpu().item())
                val_loss = np.mean(val_loss_list)
                swanlab.log({"valid_loss": val_loss})

        # save
        if iterations % config['train']['save_interval'] == 0:
            save_checkpoint(model, optimizer, iterations, CHECKPOINT_PATH)

        # log
        if iterations % config['train']['log_interval'] == 0:
            swanlab.log({"train_loss": loss.cpu().item()})
            if device.type == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                
                swanlab.log({
                    "gpu_memory_peak_GB": peak_memory
                })

if __name__ == '__main__':
    main()