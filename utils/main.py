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

## 터미널 창에 커맨드를 입력해서 스크립트를 실행할때 인자를 전달해주고 싶으면 사용합니다.
## 우리는 config.yaml 파일을 불러오기 위해서 파일 경로만 전달하면 됩니다.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='choose config file')
    return parser.parse_args()


## print() 대신에 터미널 창에 결과도 출력하고, 로그도 남기고 싶을 때 사용합니다.
## 모델 훈련 기록을 따로 남기고 싶을때 유용합니다.
def setup_logging(path):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(path),
            logging.StreamHandler()
        ]
    )


## 괴제 명세서에서 따로 언급한 collate_fn() 함수입니다.
## 데이터로더가 데이터셋으로부터 배치 단위로 데이터를 가져오는 방식을 변경할 수 있습니다.
def collate_fn(batch):
    batch = torch.stack(batch)
    return batch.permute(1, 0, 2)


## 모델의 훈련 출력값으로 로스를 계산하는 메소드입니다.
## 논문에서 제공한 공식을 그대로 구현한 코드입니다.
def loss_function(config, recon_x, x, mu, logvar, lamb, mu_att, logvar_att):
    CE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = lamb*(-0.5) * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD_attention = lamb*config['eta']*(-0.5) * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())
    return CE + KLD + KLD_attention


## 훈련을 1 에포크 수행하는 코드입니다.
## model.train(), optimizer.zero_grad(), loss.backward(), optimizer.step() 등등
## 항상 포함되는 메소드들은 용도 정도는 알고 있으면 좋습니다.
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


## 모델 평가를 1 에포크 수행하는 코드입니다.
## model.eval(), torch.no_grad() 등등
## 항상 포함되는 메소드들은 용도 정도는 알고 있으면 좋습니다.
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


## 커맨드로 스크립트 실행시 여기부터 실행됩니다.
if __name__ == '__main__':
    ## Argument parser에서 받은 값 가져옵니다.
    args = parse_args()

    ## config.yaml으로부터 사용자 설정 불러옵니다.
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    ## 결과와 로그를 저장할 디렉토리 생성합니다.
    if config['train']:
        output_dirc = './output/train_results'
    else:
        output_dirc = './output/inference_results'
    if not os.path.exists(output_dirc):
        os.makedirs(output_dirc)
    log_file_path = os.path.join(output_dirc, 'logging.log')

    ## 로깅 시작합니다.
    setup_logging(log_file_path)

    ## 실행 모드 기록합니다. 그냥 폼 잡는 용도.
    if config['train']:
        logging.info("Train mode.")
    else:
        logging.info("Inference mode.")

    ## 훈련과 인퍼런스를 진행할 디바이스 세팅합니다.
    ## 디바이스는 그냥 연산을 수행할 메인 기기라고 보시면 됩니다.
    ## CPU 개느립니다. GPU 좋을수록 속도감이 상상을 초월해요.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info(f"CUDA GPU를 사용합니다: {torch.cuda.get_device_name(0)}")
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("애플은 MPS 달려있지롱~ 개꿀!")
    else:
        device = torch.device("cpu")
        logging.info("헉.. 사용 가능한 GPU가 없어요! CPU로 열일합니다.")

    ## 데이터셋 불러옵니다.
    dataset = OutlierDataset(config)
    if config['train']:

        ## TODO 2-1: 훈련을 위한 데이터셋에 train/val 스플릿을 시켜봅시다!
        ## ----------------------------------------------------------------
        ## 1. len(dataset)을 사용하면 OutlierDataset의 __len__() 메소드를 호출합니다.
        ## 2. config에서 'split_ratio' 값을 받아옵시다.
        ## 3. train 데이터셋의 크기: int( 전체 데이터셋 길이 * split_ratio )
        ## 4. validation 데이터셋의 크기: 전체 데이터셋 길이 - train 데이터셋의 길이
        ## 5. 각 데이터셋의 크기를 구했다면, torch의 random_split 메소드를 사용해봅시다!
        ## ----------------------------------------------------------------
        "ENTER CODE HERE"
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
        ## TODO 2-2: 인퍼런스를 위한 데이터로더를 생성해봅시다!
        ## ----------------------------------------------------------------
        ## 1. 위에 제공된 train, val 데이터로더 객체들을 참고하여 인자값들을 설정해보세요.
        ## 2. 힌트는, 인퍼런스 진행 중에는 데이터의 순서를 랜덤하게 바꾸면 안된다는겁니다!
        ## ----------------------------------------------------------------
        inference_dataloader = "ENTER CODE HERE"
        ## ----------------------------------------------------------------
        logging.info("Inference 데이터셋을 성공적으로 불러왔습니다.")

    ## 모델 불러오고 사용할 디바이스에 올려줍니다.
    model = VAE(config)
    model.to(device)
    logging.info("모델을 디바이스로 불러왔습니다.")

    ## 훈련을 위한 옵티마이저, 학습률 세팅을 불러옵니다.
    ## 인퍼런스 할때는 안씁니다.
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
            ## 배치 단위 훈련 진행합니다.
            train_loss = train(lambda_kl, model, train_dataloader, device, optimizer)
            logging.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
            
            # 배치 단위 이밸류에이션 진행합니다.
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

                ## 출력 텐서 원래 순서에 맞춰서 펴줍니다. (GPU)
                data_batch = data_batch.squeeze(2).transpose(0, 1).flatten()
                output_good = output_good.squeeze(2).transpose(0, 1).flatten()

                ## 텐서를 리스트에 추가합니다.
                all_original.append(data_batch)
                all_reconstructed.append(output_good)

        ## 텐서를 한 번에 결합합니다. (GPU)
        all_original = torch.cat(all_original)
        all_reconstructed = torch.cat(all_reconstructed)

        ## NumPy 배열로 변환합니다.
        ## 어지간한 연산은 모두 텐서 상태에서 GPU로 수행하는게 좋습니다.
        ## CPU로 보내서 넘파이 배열로 연산하는 순간 시간 몇백배는 더 걸려요.
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