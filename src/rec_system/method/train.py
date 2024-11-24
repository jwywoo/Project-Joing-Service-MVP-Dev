import os
import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from rec_system.method.gmf import GMFEngine
from rec_system.method.mlp import MLPEngine
from rec_system.method.NeuMF import NeuMFEngine
from rec_system.method.data_preprocess import Loader

# 모델 설정
gmf_config = {
    'alias': 'gmf_factor8neg4-implicit',
    'num_epoch': 5,
    'batch_size': 10,
    'optimizer': 'adam',
    'adam_lr': 5e-3,
    'latent_dim': 4,
    'num_negative': 2,
    'l2_regularization': 0,
    'weight_init_gaussian': True,
    'use_cpu': True,
    'use_bachify_eval': False,
    'device_id': 0,
    'model_dir': 'output/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
}

mlp_config = {
    'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
    'num_epoch': 5,
    'batch_size': 15,
    'optimizer': 'adam',
    'adam_lr': 5e-3,
    'latent_dim': 4,
    'num_negative': 2,
    'layers': [24, 64, 32, 16, 8],
    'l2_regularization': 0.0000001,
    'weight_init_gaussian': True,
    'use_cpu': True,
    'use_bachify_eval': False,
    'device_id': 0,
    'pretrain': False,
    'model_dir': 'output/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
}

neumf_config = {
    'alias': 'neumf_factor8neg4',
    'num_epoch': 5,
    'batch_size': 10,
    'optimizer': 'adam',
    'adam_lr': 4e-3,
    'latent_dim_mf': 4,
    'latent_dim_mlp': 4,
    'num_negative': 4,
    'layers': [24, 64, 32, 16, 8],
    'l2_regularization': 0.0000001,
    'weight_init_gaussian': True,
    'use_cpu': True,
    'use_bachify_eval': True,
    'device_id': 0,
    'pretrain': False,
    'model_dir': 'output/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
}

# 데이터 경로 및 파일
file_path = 'input'
similarity_matrix_file = 'similarity_matrix.csv'

# 데이터 로드
loader = Loader(file_path, similarity_matrix_file)
train_dataset = loader.load_dataset()


# 디버깅: 데이터셋 확인 -> 삭제 필요
if train_dataset is None:
    raise ValueError("train_dataset is None. Check Loader.load_dataset().")

print(f"Train dataset loaded successfully: {len(train_dataset)} samples")


# 데이터셋 정보 기반으로 config 업데이트
common_config_updates = {
    'num_users': loader.num_users,
    'num_items': loader.num_items,
    'num_item_categories': loader.num_item_categories,  # 아이템 카테고리 수
    'num_channel_categories': loader.num_channel_categories,  # 채널 카테고리 수
    'meta_latent_dim': 4,  # 메타데이터 임베딩 차원
    'max_subscribers': 101  # 구독자 수의 최대값
}

# 설정 업데이트
gmf_config.update(common_config_updates)
mlp_config.update(common_config_updates)
neumf_config.update(common_config_updates)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=neumf_config['batch_size'], shuffle=True)

# 데이터 디버깅: 구독자 값 확인 -> 삭제 필요
for batch in train_loader:
    print(f"Subscribers in batch: {batch['subscribers']}")  # 정규화된 구독자 값
    break

# 모델 선택
model_type = 'NeuMF'  # 'GMF', 'MLP', 'NeuMF' 중 하나 선택

if model_type == 'GMF':
    engine = GMFEngine(gmf_config)
elif model_type == 'MLP':
    engine = MLPEngine(mlp_config)
elif model_type == 'NeuMF':
    engine = NeuMFEngine(neumf_config)
else:
    raise ValueError("Invalid model type. Choose 'GMF', 'MLP', or 'NeuMF'.")

# 모델 및 설정 파일 저장 디렉토리 설정
output_dir = "output"
model_dir = os.path.join(output_dir, "model")
config_dir = os.path.join(output_dir, "config")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)

# 학습 루프
for epoch in range(engine.config['num_epoch']):
    print(f"Epoch {epoch} starts!")
    print('-' * 80)

    # 모델 타입에 따라 맞는 학습 메소드 사용
    if model_type == 'MLP':
        engine.train_an_epoch_mlp(train_loader, epoch)
    elif model_type == 'NeuMF':
        engine.train_an_epoch_neumf(train_loader, epoch)
    else:
        engine.train_an_epoch(train_loader, epoch)

    # 모델 평가
    hit_ratio, ndcg = engine.evaluate(train_loader, epoch_id=epoch)

    model_filename = f"{engine.config['alias']}_Epoch{epoch}_HR{hit_ratio:.4f}_NDCG{ndcg:.4f}.model"
    model_path = os.path.join(model_dir, model_filename)

    engine.save(engine.config['alias'], epoch, hit_ratio, ndcg)

    # Config 저장 (pickle 사용)
    config_path = os.path.join(config_dir, f"config_epoch_{epoch}.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(engine.config, f)

    print(f"Model saved to: {model_path}")
    print(f"Config saved to: {config_path}")

print("\nAll models trained and evaluated successfully.")
