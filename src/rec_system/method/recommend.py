import torch
import pickle
import os
import dgl
from src.rec_system.method.model_recommend import PinSAGEModel


def load_model_and_embeddings(model_path, item_emb_path, graph_path):
    # 학습된 모델, 임베딩 및 그래프 로드
    g_list, _ = dgl.load_graphs(graph_path)
    graph = g_list[0]

    model = PinSAGEModel(graph, "item", None, hidden_dims=16, n_layers=2)
    item_emb = torch.nn.Embedding(graph.num_nodes("item"), 16, sparse=True)

    model.load_state_dict(torch.load(model_path))
    item_emb.load_state_dict(torch.load(item_emb_path))

    return model, item_emb, graph

output_dir = "./src/rec_system/model"
model_path = os.path.join(output_dir, "saved_model.pth")
item_emb_path = os.path.join(output_dir, "item_embeddings.pth")
graph_path = os.path.join(output_dir, "train_g.bin")

def is_creator_or_item(new_data):
    # 데이터의 특징에 따라 item인지 creator인지 구분하는 함수
    if 'category' in new_data and 'subscriber_count' in new_data:
        return 'creator'
    elif 'title' in new_data and 'description' in new_data:
        return 'item'
    else:
        raise ValueError("데이터가 'item'인지 'creator'인지 확인할 수 없습니다.")


def recommend_for_new_item(new_item_data, model, h_item, item_data, k=5):
    # 새로운 item의 카테고리를 기반으로 유사한 크리에이터를 찾기
    category = new_item_data.get('category')
    relevant_creators = [i for i, data in enumerate(item_data) if data['category'] == category]

    if not relevant_creators:
        print("해당 카테고리에 적합한 크리에이터가 없습니다.")
        return []

    # 카테고리 내에서 상위 k개의 추천 크리에이터 선택
    category_creator_embeddings = h_item[relevant_creators]
    new_item_embedding = model.get_repr(new_item_data, h_item)  # 새로운 아이템 임베딩 계산
    scores = torch.matmul(new_item_embedding, category_creator_embeddings.T).squeeze(0)
    top_k_scores, top_k_indices = torch.topk(scores, k)
    recommended_creators = [relevant_creators[idx] for idx in top_k_indices]
    print("추천된 상위 크리에이터:", recommended_creators)
    return recommended_creators


def recommend_for_new_creator(new_creator_data, h_item, item_data, k=5):
    # 새로운 creator의 카테고리 정보를 바탕으로 유사한 아이템을 찾기
    category = new_creator_data.get('category')
    relevant_items = [i for i, data in enumerate(item_data) if data['category'] == category]

    if not relevant_items:
        print("해당 카테고리에 적합한 아이템이 없습니다.")
        return []

    # 카테고리 내에서 상위 k개의 추천 아이템 선택
    category_item_embeddings = h_item[relevant_items]
    scores = torch.sum(category_item_embeddings, dim=0)
    top_k_scores, top_k_indices = torch.topk(scores, k)
    recommended_items = [relevant_items[idx] for idx in top_k_indices]
    print("추천된 상위 아이템:", recommended_items)
    return recommended_items


def recommend_based_on_new_data(new_data, model, h_item, item_data):
    data_type = is_creator_or_item(new_data)

    if data_type == 'item':
        return recommend_for_new_item(new_data, model, h_item, k=5)
    elif data_type == 'creator':
        return recommend_for_new_creator(new_data, h_item, item_data, k=5)


