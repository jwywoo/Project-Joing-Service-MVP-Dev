import torch
import pickle
from rec_system.method.NeuMF import NeuMF
from rec_system.method.data_preprocess import Loader, TextEmbedder


# 추천 시스템 클래스
class Recommender:
    def __init__(self, model_path, config_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(config_path, "rb") as f:
            self.config = pickle.load(f)

        print("로드된 Config 확인:", self.config)

        self.model = NeuMF(self.config).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        file_path = 'src/rec_system/model/input'
        similarity_matrix_file = 'src/rec_system/model/similarity_matrix.csv'

        self.loader = Loader(file_path, similarity_matrix_file)
        self.text_embedder = TextEmbedder()

        # 사용자 및 아이템 메타데이터 로드
        self.user_metadata = self.loader.load_user_metadata()
        self.item_metadata = self.loader.load_item_metadata()

    def preprocess_new_item(self, data):
        """
        새로운 아이템 데이터 전처리
        """
        data = {
            'item_id': self.config['num_items'] - 1,  # 범위를 초과하지 않도록 수정
            'item_category': self.loader.similarity_matrix.columns.tolist().index(data['item_category']),
            'media_type': 0 if 'short' in data['media_type'].lower() else 1,
            'item_embedding': torch.tensor(self.text_embedder.get_text_embedding(data['title']), dtype=torch.float),
            'subscribers': 0,  # 아이템에 대해서는 구독자 수를 0으로 설정
            'channel_category': 0,  # 기본 채널 카테고리 값
            'creator_embedding': torch.zeros(768)  # BERT 임베딩 차원이 768이라고 가정
        }
        return data

    def preprocess_new_creator(self, data):
        """
        새로운 크리에이터 데이터 전처리
        """
        normalized_subscribers = self.loader.normalize_subscribers(
            data['subscribers'], self.config['max_subscribers']
        )

        data = {
            'creator_id': self.config['num_users'] - 1,  # 범위를 초과하지 않도록 수정
            'channel_category': self.loader.similarity_matrix.columns.tolist().index(data['channel_category']),
            'creator_embedding': torch.tensor(self.text_embedder.get_text_embedding(data['channel_name']), dtype=torch.float),
            'subscribers': normalized_subscribers,
            'item_category': 0,
            'media_type': 0,
            'item_embedding': torch.zeros(384)
        }
        return data

    def recommend_for_new_item(self, item_data, top_k=10):
        """
        새로운 아이템 데이터에 대해 사용자 추천
        """
        item_data = self.preprocess_new_item(item_data)
        user_ids_tensor = torch.arange(self.config['num_users'], dtype=torch.long).to(self.device)
        item_id_tensor = torch.tensor([item_data['item_id']], dtype=torch.long).to(self.device)
        item_category_tensor = torch.tensor([item_data['item_category']], dtype=torch.long).to(self.device)
        media_type_tensor = torch.tensor([item_data['media_type']], dtype=torch.long).to(self.device)

        # 채널 카테고리와 구독자 수를 기본값으로 설정
        channel_category_tensor = torch.zeros(1, dtype=torch.long).to(self.device)  # 기본값 0
        subscribers_tensor = torch.zeros(1, dtype=torch.long).to(self.device)  # 기본값 0

        with torch.no_grad():
            scores = self.model(
                user_ids_tensor,
                item_id_tensor.repeat(self.config['num_users']),
                item_category_tensor.repeat(self.config['num_users']),
                media_type_tensor.repeat(self.config['num_users']),
                channel_category_tensor.repeat(self.config['num_users']),
                subscribers_tensor.repeat(self.config['num_users']),
            )
            scores = scores.view(-1).cpu().numpy()

        top_k_indices = scores.argsort()[-top_k:][::-1].copy()
        recommended_user_ids = user_ids_tensor[top_k_indices].cpu().numpy()

        # 사용자 메타데이터에서 추천된 사용자 정보를 가져옴
        recommended_creator_data = []
        for user_id in recommended_user_ids:
            user_metadata = self.user_metadata[user_id]
            recommended_creator_data.append({
                'creator_id': int(user_id),
                'channel_category': user_metadata['channel_category'],
                'channel_name': user_metadata['channel_name'],
                'subscribers': user_metadata['subscribers']
            })

        return recommended_creator_data

    def recommend_for_new_creator(self, creator_data, top_k=10):
        """
        새로운 크리에이터 데이터에 대해 아이템 추천
        """
        creator_data = self.preprocess_new_creator(creator_data)

        # channel_category 범위 초과 값 처리
        max_channel_category = self.config['num_channel_categories'] - 1
        creator_data['channel_category'] = min(creator_data['channel_category'], max_channel_category)

        user_id_tensor = torch.tensor([creator_data['creator_id']], dtype=torch.long).to(self.device)
        item_ids_tensor = torch.arange(self.config['num_items'], dtype=torch.long).to(self.device)

        with torch.no_grad():
            scores = self.model(
                user_id_tensor.repeat(self.config['num_items']),
                item_ids_tensor,
                torch.tensor([creator_data['item_category']], dtype=torch.long).repeat(self.config['num_items']).to(
                    self.device),
                torch.tensor([creator_data['media_type']], dtype=torch.long).repeat(self.config['num_items']).to(
                    self.device),
                torch.tensor([creator_data['channel_category']], dtype=torch.long).repeat(self.config['num_items']).to(
                    self.device),
                torch.tensor([creator_data['subscribers']], dtype=torch.long).repeat(self.config['num_items']).to(
                    self.device)
            )
            scores = scores.view(-1).cpu().numpy()

        top_k_indices = scores.argsort()[-top_k:][::-1].copy()
        recommended_items = item_ids_tensor[top_k_indices].cpu().numpy()

        # 아이템 메타데이터에서 추천된 아이템 정보를 가져옴
        recommended_item_data = []
        for item_id in recommended_items:
            item_metadata = self.item_metadata[item_id]
            recommended_item_data.append({
                'item_id': int(item_id),
                'title': item_metadata['title'],
                'item_category': item_metadata['item_category'],
                'media_type': item_metadata['media_type'],
                'score': item_metadata['score'],
                'item_content': item_metadata['item_content']
            })

        return recommended_item_data


if __name__ == "__main__":
    # 저장된 모델과 config 경로
    model_path = "output/neumf_factor8neg4_Epoch4_HR1.0000_NDCG1.0000.model"
    config_path = "output/config/config_epoch_4.pkl"

    # Recommender 초기화
    recommender = Recommender(model_path, config_path)

    # 새로운 아이템 데이터 예시
    new_item_data = {
        'title': "바밤바를 뛰어넘는 밤 맛 과자가 있을까?",
        'item_category': 'entertainment',
        'media_type': 'short',
        'score': 80,
        'item_content': '다양한 밤 맛 과자를 비교하며 맛과 질감을 리뷰하는 콘텐츠'
    }

    # 아이템에 대한 사용자 추천
    recommended_users = recommender.recommend_for_new_item(new_item_data)
    print(f"추천 사용자 목록: {recommended_users}")

    # 새로운 creator 데이터 예시
    new_creator_data = {
        'channel_category': "tech",
        'channel_name': "최마태의 POST IT",
        'subscribers': 263000
    }

    # 크리에이터에 대한 아이템 추천
    recommended_items = recommender.recommend_for_new_creator(new_creator_data)
    print(f"추천 아이템 목록: {recommended_items}")
