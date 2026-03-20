import os
import numpy as np
from matplotlib import pyplot as plt
from models.SWINCrossattention import SwinIR
import warnings
from utils.model_utils import normalize_to_minus_one_one
warnings.simplefilter("ignore")
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.dataset import MRIDataset
from utils.config_loader import load_config
from utils.logger import setup_logger
from utils.model_utils import set_seed, save_model,load_model
from torchvision import transforms


def add_salt_and_pepper_noise(tensor, prob=0.1):
    """
    Add salt and pepper noise to a tensor.

    :param tensor: Input tensor
    :param prob: Probability of noise
    :return: Noisy tensor
    """
    noisy_tensor = tensor.clone()
    num_elements = torch.numel(noisy_tensor)
    num_noise = int(num_elements * prob)

    # Salt noise
    salt_indices = torch.randint(0, num_elements, (num_noise,))
    noisy_tensor.view(-1)[salt_indices] = torch.max(tensor)

    # Pepper noise
    pepper_indices = torch.randint(0, num_elements, (num_noise,))
    noisy_tensor.view(-1)[pepper_indices] = torch.min(tensor)

    return noisy_tensor.to(torch.float32)

def add_gaussian_noise(image, prob=0.02):
    img = np.array(image)

    noisy_img = img.copy()
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                noisy_img[i][j] = 0  # 椒噪声
            elif rdn > thres:
                noisy_img[i][j] = 255  # 盐噪声

    tensor = torch.from_numpy(noisy_img)
    return tensor.to(torch.float32)

def data_maker(config_path):
        # 加载配置
        base_config_path = os.path.join(os.path.dirname(config_path), "base.yaml")
        config = load_config(config_path, base_config_path)

        # 设置随机种子
        set_seed(config['seed'])

        # 创建目录
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)

        # 设置日志
        logger = setup_logger("TokenizerTrainer", config['log_dir'])
        writer = SummaryWriter(log_dir=config['log_dir'])

        # 记录配置
        logger.info(f"Starting Tokenizer training with config:\n{config}")

        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"运行设备: {device}")
        logger.info(f"Using device: {device}")

        # 创建数据集
        dataset_train = MRIDataset(
            root_dir=config['data_dir'],
            mode='all', Flag='train',
            json_path=config['test__mapping_file'],
        )
        dataset_test = MRIDataset(
            root_dir=config['data_dir'],
            mode='all', Flag='test',
            json_path = config['test__mapping_file'],
        )
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )

        # 实例化 SWinIR
        return dataloader_train,dataloader_test,config

def show(recon_img,images):
    rec_image = recon_img.cpu()
    rec_image = rec_image.detach().numpy()
    images = images.cpu().numpy()
    for i in range(1, rec_image.shape[0]):
        im2 = np.zeros((256, 256, 3))
        im = np.zeros((256, 256, 3))
        im[:, :, 2] = rec_image[i, 2, :, :]
        im[:, :, 1] = rec_image[i, 1, :, :]
        im[:, :, 0] = rec_image[i, 0, :, :]
        im2[:, :, 2] = images[i, 2, :, :]
        im2[:, :, 1] = images[i, 1, :, :]
        im2[:, :, 0] = images[i, 0, :, :]
        im[0, 0, 0] = 0
        im[1, 0, 0] = 1
        im2[0, 0, 0] = 0
        im2[1, 0, 0] = 1
        im = normalize_to_minus_one_one(im)
        im2 = normalize_to_minus_one_one(im2)
        plt.clf()
        plt.imshow(im)
        plt.pause(0.4)
        plt.imshow(im2)
        plt.pause(0.5)
    plt.show()


def train(dataloader_train,dataloader_test,config):
    upscale = 1  # 图像放大因子
    window_size = 1  # 窗口大小
    height = config['image_size']
    width = config['image_size']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("TokenizerTrainer", config['log_dir'])
    writer = SummaryWriter(log_dir=config['log_dir'])
    model = SwinIR(
        upscale=upscale,            img_size=(height, width),   patch_size=config['patch_size'],
        window_size=window_size,    img_range=1.,               depths=[8, 8, 8, 8],
        in_chans=44,                embed_dim=config['dim'],    num_heads=[6, 6, 6, 6],
        mlp_ratio=2,                upsampler='',               latent_dim=config['latent_dim'])
    # load_model(model, os.path.join(config['save_dir'], "tokenizer_final.pt"))
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    # 训练循环
    global_step = 0
    for epoch in range(config['epochs']):
        # ===== 训练阶段 =====
        model.train()
        model.to(device)
        total_loss = 0.0
        for batch_idx, (images, test,_) in enumerate(dataloader_train):
            images = images.to(device)
            images = normalize_to_minus_one_one(images)

            # 前向传播
            recon_img, vq_loss = model(images,test)
            recon_loss = nn.MSELoss()(recon_img, images)
            recon_loss = vq_loss + recon_loss

            # 反向传播
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

            total_loss += recon_loss.item()
            global_step += 1

            # 记录训练损失
            if batch_idx % config['log_interval'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(
                    f"Epoch {epoch + 1}/{config['epochs']} | Batch {batch_idx}/{len(dataloader_train)} | Loss: {avg_loss:.4f}")
                writer.add_scalar('Loss/train', recon_loss.item(), global_step)

        # ===== 测试阶段 =====
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for test_idx, (test_images,test,_) in enumerate(dataloader_test):
                test_images = test_images.to(device)
                test_images = normalize_to_minus_one_one(test_images)

                # 测试前向传播
                test_recon, test_vq_loss = model(test_images)
                test_recon_loss = nn.MSELoss()(test_recon, test_images)
                test_total_loss = test_vq_loss + test_recon_loss

                test_loss += test_total_loss.item()

        # 计算平均测试损失
        avg_test_loss = test_loss / len(dataloader_test)
        logger.info(
            f"Epoch {epoch + 1}/{config['epochs']} | Test Loss: {avg_test_loss:.4f}")
        writer.add_scalar('Loss/test', avg_test_loss, epoch)

        # ===== 保存检查点 =====
        if (epoch + 1) % config['save_interval'] == 0 or (epoch + 1) == config['epochs']:
            checkpoint_path = os.path.join(config['save_dir'], f"tokenizer_epoch_{epoch + 1}.pt")
            save_model(model, checkpoint_path, epoch=epoch, optimizer=optimizer)




        # 保存检查点
        if (epoch + 1) % config['save_interval'] == 0 or (epoch + 1) == config['epochs']:
            # show(recon_img, images)
            checkpoint_path = os.path.join(config['save_dir'], f"tokenizer_epoch_{epoch + 1}.pt")
            save_model(model, checkpoint_path, epoch=epoch, optimizer=optimizer)

    # 保存最终模型
    final_path = os.path.join(config['save_dir'], "tokenizer_final.pt")
    save_model(model, final_path)

    writer.close()
    logger.info("Tokenizer training completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Tokenizer model')
    parser.add_argument('--config', type=str, default='./configs/tokenizer.yaml',
                        help='Path to config file')

    args = parser.parse_args()
    dataloader_train,dataloader_test,config = data_maker(args.config)
    train(dataloader_train,dataloader_test,config)