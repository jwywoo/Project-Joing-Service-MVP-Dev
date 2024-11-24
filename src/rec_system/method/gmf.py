import torch
from rec_system.method.engine import Engine
from rec_system.method.utils import use_cpu
from torch import nn


class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(GMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim

        # 임베딩 레이어
        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # 출력 레이어
        self.affine_output = nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)  # 사용자 임베딩
        item_embedding = self.embedding_item(item_indices)  # 아이템 임베딩
        element_product = torch.mul(user_embedding, item_embedding)  # 내적 계산 (element-wise multiplication)
        logits = self.affine_output(element_product)  # 선형 변환
        rating = self.logistic(logits)  # 시그모이드 함수
        return rating


class GMFEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        self.model = GMF(config['num_users'], config['num_items'], config['latent_dim'])

        # CPU 환경에서 모델을 실행하도록 설정
        if config['use_cpu'] is True:
            self.model.to(use_cpu())  # Use the custom use_cpu method to force CPU usage

        super(GMFEngine, self).__init__(config)
        print(self.model)
