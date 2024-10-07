import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class OutlierDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = self.prepare_data()

    def __len__(self) -> int:
        """
        - 생성한 데이터셋 클래스에 들어있는 샘플의 수를 리턴합니다.
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        - 입력한 인덱스에 위치한 데이터 샘플을 꺼내옵니다.
        """
        return self.data[idx]

    def load_csv(self, data_path: str) -> np.array:
        """
        - 입력받은 경로에 있는 csv파일을 데이터프레임으로 받습니다.
        - 데이터프레임에서 'value'열에 있는 값들만 넘파이 배열로 반환합니다.
        """
        try:
            data = pd.read_csv(data_path)
            data = data['value'].values
        except Exception as e:
            print(f"csv 파일을 불러오는 도중 문제가 발생했습니다: {e}")
            print("config.yaml을 열어 csv 파일 경로가 상대경로로 기입되어 있는지 확인하세요.")
        return data

    def slice_by_window(self, data: np.array, window_size: int, step_size: int) -> np.ndarray:
        """
        - window_size크기의 윈도우를 step_size만큼 이동하며 데이터를 분할합니다.
        - 참고로 window_size는 biLSTM 인코더의 입력 차원과 같아야합니다.
        """
        try:
            if window_size <= 0 or step_size <= 0:
                raise ValueError("window_size와 step_size는 양의 정수값이어야 합니다.")
            if window_size > len(data):
                raise ValueError("window_size는 주어진 데이터셋의 길이보다 같거나 작아야합니다.")

            ## TODO 1: slice_by_window 메소드의 핵심 코드를 완성해보세요!
            ## ----------------------------------------------------------------
            ## 1. 빈 list를 하나 생성합니다.
            ## 2. 각 분할구간의 시작점은 인덱스 0부터 step_size 단위로 증가합니다.
            ## 3. 각 분할구간의 종료점은 인덱스 window_size-1부터 step_size 단위로 증가합니다.
            ## 4. 위 정보를 고려하여 range 함수를 써서 루프 범위를 지정해줍니다.
            ## 5. 이제 빈 리스트에 각 분할구간의 값들을 순차적으로 담아주면 되겠네요!
            ## ----------------------------------------------------------------
            windows = []
            for start in range(0, len(data) - window_size + 1, step_size):
                window = data[start:start + window_size]
                windows.append(window)
            ## ----------------------------------------------------------------

        except ValueError as e:
            print(f"데이터를 윈도우 단위로 슬라이싱 하는 도중 문제가 발생했습니다: {e}")

        return np.array(windows)

    def standardize(self, data: np.ndarray) -> np.ndarray:
        """
        - 모든 값들을 시퀀스 단위로 정규화합니다.
        """

        ## TODO 2: 입력 데이터의 구조를 감안하여 윈도우 단위의 정규화 코드를 완성해보세요!
        ## ----------------------------------------------------------------
        ## 1. 앞부분이 잘 구현됐다면, 현재 입력값의 차원은 [총 샘플 수, 윈도우 크기] 형태입니다.
        ## 2. 그렇다면 axis=1 상으로 평균과 표준편차를 계산해줄 수 있겠네요.
        ## 3. keepdims 인자가 어떤 역할을 하는지 알아보시기 바랍니다.
        ## 4. 표준편차가 재수없게 0인 케이스는 전부 1로 바꿔서 연산을 진행하도록 처리해주세요!
        ## 5. 마지막으로 정규화된 데이터를 뽑아주세요.
        ## ----------------------------------------------------------------
        means = data.mean(axis=1, keepdims=True)
        stds = data.std(axis=1, keepdims=True)
        stds = np.where(stds == 0, 1, stds)
        normalized_data = (data - means) / stds
        ## ----------------------------------------------------------------

        return normalized_data
    
    def to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """
        - 정규화한 데이터를 [총 샘플 수, 윈도우 크기, 1] 차원의 텐서로 반환합니다.
        """
        ## TODO 3: 입력 데이터를 [총 샘플 수, 윈도우 크기, 1]로 반환해보세요!
        ## ----------------------------------------------------------------
        ## 1. 현재 입력값은 [총 샘플 수, 윈도우 크기] 차원의 넘파이 다차원 배열입니다.
        ## 2. 데이터를 GPU에 올리기 위해 토치 텐서로 변환해야합니다. (torch.tensor)
        ## 3. 모델에게 올바른 입력값을 제공하기 위해 차원도 바꿔줘야합니다. torch.unsqueeze)
        ## ----------------------------------------------------------------
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)
        ## ----------------------------------------------------------------
        return tensor

    def add_noise(self, data: torch.Tensor, std: int) -> torch.Tensor:
        """
        - 모든 값에 평균을 0, 표준편차를 std로 하는 정규분포로부터 샘플링된 노이즈를 추가합니다.
        """
        ## TODO 4: 입력 데이터의 차원에 맞춰 노이즈 값들을 추가해보세요!
        ## ----------------------------------------------------------------
        ## 1. 정규분포를 텐서 타입으로 생성하는 메소드를 써보세요! (torch.normal)
        ## 2. 주어진 데이터와 똑같은 차원을 가지는 노이즈를 만드러면 어떤 인자를 전달해야할지 알아보세요!
        ## ----------------------------------------------------------------
        noise = torch.normal(0, std, data.size())
        ## ----------------------------------------------------------------
        return data + noise

    def prepare_data(self) -> torch.Tensor:
        """
        - 훈련, 테스트 케이스에 따라서 데이터 전처리를 개별적으로 진행합니다.
        """
        data_path = self.config['data_path']
        seq_size = self.config['seq_size']
        step_size = self.config['step_size']
        noise_std = self.config['noise_std']
        train = self.config['train']

        if train:
            data = self.load_csv(data_path)
            data = self.slice_by_window(data, seq_size, step_size)
            data = self.standardize(data)
            data = self.to_tensor(data)
            data = self.add_noise(data, noise_std)
        else:
            data = self.load_csv(data_path)
            data = self.slice_by_window(data, seq_size, seq_size)
            data = self.standardize(data)
            data = self.to_tensor(data)
        return data