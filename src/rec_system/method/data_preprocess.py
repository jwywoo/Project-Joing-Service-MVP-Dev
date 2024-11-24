import pickle
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
import numpy as np
from sentence_transformers import SentenceTransformer

# BERT 임베딩을 위한 클래스
class TextEmbedder:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def get_text_embedding(self, text):
        if not text or text.strip() == "":  # 빈 문자열 또는 None 체크
            print(f"Warning: Empty text encountered. Text value: '{text}'")
            raise ValueError("Text input is empty or invalid. Please check the data source.")  # 예외 발생

        try:
            embedding = self.model.encode(text)
        except Exception as e:  # Sentence-BERT에서 예외 발생 시 처리
            print(f"Error encoding text '{text}': {e}")
            raise e  # 예외를 다시 발생시켜 문제를 명확히 표시

        return embedding


# Dataset 클래스 정의
class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor, item_titles, creator_names,
                 item_category, media_type, channel_category, subscribers, item_category_similarities):
        self.user_tensor = torch.tensor(user_tensor, dtype=torch.long)
        self.item_tensor = torch.tensor(item_tensor, dtype=torch.long)
        self.target_tensor = torch.tensor(target_tensor, dtype=torch.float)
        self.item_titles = item_titles
        self.creator_names = creator_names
        self.item_category = item_category
        self.media_type = media_type
        self.channel_category = channel_category
        self.subscribers = subscribers
        self.item_category_similarities = torch.tensor(item_category_similarities, dtype=torch.float)
        self.text_embedder = TextEmbedder()
        # 임베딩 계산
        self.item_embeddings = [self.text_embedder.get_text_embedding(title) for title in self.item_titles]
        self.creator_embeddings = [self.text_embedder.get_text_embedding(name) for name in self.creator_names]

    def __len__(self):
        return len(self.user_tensor)

    def __getitem__(self, idx):
        item_embedding = self.item_embeddings[idx]
        creator_embedding = self.creator_embeddings[idx]
        item_category_similarity = torch.tensor(self.item_category_similarities[idx], dtype=torch.float).unsqueeze(0)

        return {
            'user_id': self.user_tensor[idx],
            'item_id': self.item_tensor[idx],
            'target': self.target_tensor[idx],
            'item_category': torch.tensor(self.item_category[idx], dtype=torch.long),
            'media_type': torch.tensor(self.media_type[idx], dtype=torch.long),
            'channel_category': torch.tensor(self.channel_category[idx], dtype=torch.long),
            'subscribers': torch.tensor(self.subscribers[idx], dtype=torch.long),
            'item_embedding': torch.tensor(item_embedding, dtype=torch.float),
            'creator_embedding': torch.tensor(creator_embedding, dtype=torch.float),
            'item_category_similarity': item_category_similarity,
        }


class Loader:
    def __init__(self, file_path, similarity_matrix_file):
        self.file_path = file_path
        self.similarity_matrix_file = similarity_matrix_file
        self.similarity_matrix = self.load_similarity_matrix()
        self.text_embedder = TextEmbedder()

        # 변수 초기화
        self.num_users = None
        self.num_items = None
        self.num_item_categories = None
        self.num_channel_categories = None
        self.max_subscribers = None

    def load_similarity_matrix(self):
        return pd.read_csv(self.similarity_matrix_file, index_col=0)

    def normalize_subscribers(self, subscribers, max_value, scale=100):
        """
        구독자 수를 0~scale 범위로 정규화.
        """
        normalized = np.round((subscribers / max_value) * scale).astype(int)
        return np.clip(normalized, 0, scale)

    def load_dataset(self):
        # 파일 로드
        item_df = pd.read_csv(self.file_path + '/Item_random25.csv')
        creator_df = pd.read_csv(self.file_path + '/Creator_random25.csv')

        # 사용자와 아이템 매핑
        item_mapping = {item_id: idx for idx, item_id in enumerate(item_df['item_id'].unique())}
        creator_mapping = {creator_id: idx for idx, creator_id in enumerate(creator_df['creator_id'].unique())}

        # 매핑된 ID로 변환
        item_df['item_id'] = item_df['item_id'].map(item_mapping)
        creator_df['creator_id'] = creator_df['creator_id'].map(creator_mapping)

        # 데이터 전처리
        item_df['item_category'] = item_df['item_category'].astype("category").cat.codes
        item_df['media_type'] = item_df['media_type'].map({'short': 0, 'long': 1})
        item_df['target'] = item_df['score'].apply(lambda x: 1 if x >= 0.85 else 0)  # 0과 1로 설정

        creator_df['channel_category'] = creator_df['channel_category'].astype("category").cat.codes
        creator_df['subscribers'] = creator_df['subscribers'].replace({',': ''}, regex=True).astype(int)

        # 최대 구독자 수 설정 및 정규화
        fixed_max_value = 10000000  # 고정된 최대값
        print(f"Original subscribers: {creator_df['subscribers'].head()}")  # 디버깅: 원본 값 확인
        creator_df['subscribers'] = self.normalize_subscribers(
            creator_df['subscribers'].astype(float), fixed_max_value
        )

        print(f"Normalized subscribers: {creator_df['subscribers'].head()}")  # 디버깅: 정규화된 값 확인

        # 최대 구독자 수를 임베딩에 활용할 수 있도록 설정
        self.num_users = creator_df['creator_id'].nunique()
        self.num_items = item_df['item_id'].nunique()
        self.num_item_categories = item_df['item_category'].nunique()
        self.num_channel_categories = creator_df['channel_category'].nunique()
        self.max_subscribers = creator_df['subscribers'].max()

        # 아이템 카테고리 유사도
        item_category_similarities = item_df['item_category'].apply(self.calculate_category_similarity).values

        # 데이터셋 객체 반환
        return UserItemRatingDataset(
            user_tensor=item_df['item_id'].values,
            item_tensor=creator_df['creator_id'].values,
            target_tensor=item_df['target'].values,
            item_titles=item_df['title'].values,
            creator_names=creator_df['channel_name'].values,
            item_category=item_df['item_category'].values,
            media_type=item_df['media_type'].values,
            channel_category=creator_df['channel_category'].values,
            subscribers=creator_df['subscribers'].values,  # 원본 값 사용
            item_category_similarities=item_category_similarities
        )

    def load_user_metadata(self):
        """
        사용자 메타데이터 로드
        """
        user_metadata_file = f"{self.file_path}/Creator_random25.csv"  # 사용자 데이터 경로
        user_metadata = pd.read_csv(user_metadata_file)
        user_metadata_dict = user_metadata.to_dict('index')
        return user_metadata_dict

    def load_item_metadata(self):
        """
        아이템 메타데이터 로드
        """
        item_metadata_file = f"{self.file_path}/Item_random25.csv"  # 아이템 데이터 경로
        item_metadata = pd.read_csv(item_metadata_file)
        item_metadata_dict = item_metadata.to_dict('index')
        return item_metadata_dict

    def get_meta_info(self):
        """모든 메타 정보를 반환"""
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_item_categories': self.num_item_categories,
            'num_channel_categories': self.num_channel_categories,
            'max_subscribers': self.max_subscribers,
        }

    def calculate_category_similarity(self, category_code):
        if category_code in self.similarity_matrix.columns:
            return self.similarity_matrix.loc[category_code, category_code]
        return 0.5
