import os
import yaml
import logging
import platform
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.set_loglevel('WARNING')

import torch
from torch import optim
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from model import VAE
from dataset import OutlierDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='choose config file')
    return parser.parse_args()

def setup_logging(path):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(path),
            logging.StreamHandler()
        ]
    )

def collate_fn(batch):
    batch = torch.stack(batch)
    return batch.permute(1, 0, 2)

def loss_function(config, recon_x, x, mu, logvar, lamb, mu_att, logvar_att):
    CE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = lamb*(-0.5) * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD_attention = lamb*config['eta']*(-0.5) * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())
    return CE + KLD + KLD_attention

def train(lambda_kl: float,
          model: VAE,
          dataloader: DataLoader,
          device: torch.device,
          optimizer: optim.Optimizer) -> float:
    
    model.train()
    train_loss = 0
    
    for trainbatch in dataloader:
        trainbatch = trainbatch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, _, _, _, _, _, c_t_mus, c_t_logvars, _ = model(trainbatch)
        loss = loss_function(config, recon_batch, trainbatch, mu, logvar, lambda_kl, c_t_mus, c_t_logvars)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(dataloader)
    
def test(lambda_kl: float,
         model: VAE,
         dataloader: DataLoader,
         device: torch.device) -> float:
    
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for testbatch in dataloader:
            testbatch = testbatch.to(device)
            recon_batch, mu, logvar, _, _, _, _, _, c_t_mus, c_t_logvars, _, _ = model(testbatch)
            loss = loss_function(config, recon_batch, testbatch, mu, logvar, lambda_kl, c_t_mus, c_t_logvars)
            test_loss += loss.item()
    return test_loss / len(dataloader)


if __name__ == '__main__':
    ## Argument Parser
    args = parse_args()

    ## config.yaml으로부터 사용자 설정 불러오기.
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    ## 결과와 로그를 저장할 디렉토리 생성.
    if config['train']:
        output_dirc = './output/train_results'
    else:
        output_dirc = './output/inference_results'
    if not os.path.exists(output_dirc):
        os.makedirs(output_dirc)
    log_file_path = os.path.join(output_dirc, 'logging.log')

    ## 로깅 시작하기.
    setup_logging(log_file_path)

    ## 실행 모드 기록하기.
    if config['train']:
        logging.info("Train mode.")
    else:
        logging.info("Inference mode.")

    ## 훈련과 인퍼런스를 진행할 디바이스 세팅하기.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info(f"CUDA GPU를 사용합니다: {torch.cuda.get_device_name(0)}")
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("애플은 MPS 달려있지롱~ 개꿀!")
    else:
        device = torch.device("cpu")
        logging.info("헉.. 사용 가능한 GPU가 없어요! CPU로 열일합니다.")

    ## 데이터셋 불러오기.
    dataset = OutlierDataset(config)
    if config['train']:

        ## TODO 5: 훈련을 위한 데이터셋에 train/val 스플릿을 시켜봅시다!
        ## ----------------------------------------------------------------
        ## 1. len(dataset)을 사용하면 OutlierDataset의 __len__() 메소드를 호출합니다.
        ## 2. config에서 'split_ratio' 값을 받아옵시다.
        ## 3. train 데이터셋의 크기: int( 전체 데이터셋 길이 * split_ratio )
        ## 4. validation 데이터셋의 크기: 전체 데이터셋 길이 - train 데이터셋의 길이
        ## 5. 각 데이터셋의 크기를 구했다면, torch의 random_split 메소드를 사용해봅시다!
        ## ----------------------------------------------------------------
        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.8)
        validation_size = dataset_size - train_size
        train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
        ## ----------------------------------------------------------------

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=config['batch_size'],
                                      shuffle=True,
                                      drop_last = True,
                                      collate_fn=collate_fn)
        val_dataloader = DataLoader(validation_dataset,
                                    batch_size=config['batch_size'],
                                    shuffle=False,
                                    drop_last = True,
                                    collate_fn=collate_fn)
        logging.info("Training/validation 데이터셋을 성공적으로 불러왔습니다.")
    else:
        ## TODO 6: 인퍼런스를 위한 데이터로더를 생성해봅시다!
        ## ----------------------------------------------------------------
        ## 1. 위에 제공된 train, val 데이터로더 객체들을 참고하여 인자값들을 설정해보세요.
        ## 2. 힌트는, 인퍼런스 진행 중에는 데이터의 순서를 랜덤허게 바꾸면 안된다는겁니다!
        ## 3. 참고로 collate_fn 인자의 역할은 데이터로더가 데이터셋으로부터 배치 단위로
        ##    값들을 가져올 때, 우리가 그 방식에 관여하고 싶다면 메소드로 설정해주는 것입니다.
        ##    사실 우리가 사용할 모델은 입력값을 [seq_size, batch_size, input_features=1]
        ##    형태로 받고 있는데요, 우리는 데이터셋 클래스를 만들때 [num_samples, seq_size,
        ##    input_features=1]차원으로 데이터셋을 저장하게 했었죠. 따라서 collate_fn을
        ##    설정해주지 않은 데이터로더로 값을 부르면, 실제로는 [batch_size, seq_size,
        ##    input_features=1]차원으로 값을 가져오게 됩니다. 이걸 모델의 입력 형태에 맞춰주기
        ##    위해서 추가적으로 메소드를 적용시켜준 것이라고 이해하시면 되겠습니다.
        ## ----------------------------------------------------------------
        inference_dataloader = DataLoader(dataset.data,
                                          batch_size=config['batch_size'],
                                          shuffle=False,
                                          drop_last = True,
                                          collate_fn = collate_fn)
        ## ----------------------------------------------------------------
        logging.info("Inference 데이터셋을 성공적으로 불러왔습니다.")

    ## 모델 불러오고 사용할 디바이스에 올려주기.
    model = VAE(config)
    model.to(device)
    logging.info("모델을 디바이스로 불러왔습니다.")

    ## 훈련을 위한 옵티마이저, 학습률 세팅을 불러오기.
    if config['train']:
        optimizer_choice = config['optimizer_choice']
        learning_rate = config['learning_rate']

        if optimizer_choice == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            logging.info("Using AdamW optimizer.")
        elif optimizer_choice == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            logging.info("Using SGD optimizer.")
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            logging.info("Using Adam optimizer.")

        logging.info(f"Optimizer choice: {optimizer_choice}, Learning rate: {learning_rate}")

    ## 훈련을 시작해봅시다.
    if config['train']:
        logging.info("모델 훈련을 시작합니다.")
        epochs = config['epochs']
        lambda_kl = config['lambda_kl']

        for epoch in range(1, epochs + 1):
            ## 배치 단위 훈련 진행
            train_loss = train(lambda_kl, model, train_dataloader, device, optimizer)
            logging.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
            
            # 배치 단위 이밸류에이션 진행.
            val_loss = test(lambda_kl, model, val_dataloader, device)
            logging.info(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")
            
        # 훈련이 종료되면 모델 체크포인트, 즉 파라미터 가중치들을 전부 담은 .pt 파일을 아웃풋 디렉토리에 저장합니다.
        checkpoint_path = os.path.join(output_dirc, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Model checkpoint saved at epoch {epoch} to {checkpoint_path}")
        
        # 훈련을 돌리는데 사용한 설정값들도 같은 아웃풋 디렉토리에 저장해둡시다. 나중에 궁금할지도 모르잖아요?
        last_config_path = os.path.join(output_dirc, 'last_config.yaml')
        with open(last_config_path, 'w') as file:
            yaml.dump(config, file)
        logging.info(f"User configurations saved to {last_config_path}")
        logging.info("Training completed successfully.")
    
    ## 인퍼런스를 시작해봅시다.
    else:
        logging.info("인퍼런스를 시작합니다.")
        all_original = []
        all_reconstructed = []

        with torch.no_grad():
            for data_batch in inference_dataloader:
                data_batch = data_batch.to(device)
                output_good, mu, logvar, _, _, _, _, _, _, _, _, _ = model(data_batch)

                ## 데이터 처리 (CPU 변환 없이 GPU 상에서 작업)
                data_batch = data_batch.squeeze(2).transpose(0, 1).flatten()
                output_good = output_good.squeeze(2).transpose(0, 1).flatten()

                ## 텐서 리스트에 추가
                all_original.append(data_batch)
                all_reconstructed.append(output_good)

        ## 텐서를 한 번에 결합 (GPU에서 수행)
        all_original = torch.cat(all_original)
        all_reconstructed = torch.cat(all_reconstructed)

        ## NumPy 배열로 변환
        all_original_np = all_original.cpu().numpy()
        all_reconstructed_np = all_reconstructed.cpu().numpy()

        ## 그림 그려보자잇
        plot_path = os.path.join(output_dirc, 'original_vs_reconstructed.png')
        plt.figure(figsize=(16, 5))
        plt.plot(all_original_np, label='Original', alpha=1, color='blue')
        plt.plot(all_reconstructed_np, label='Reconstructed', alpha=0.5, color='orange')
        plt.title('Original VS Reconstructed Data')
        plt.legend()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logging.info("인퍼런스 시각화 결과를 저장했습니다.")