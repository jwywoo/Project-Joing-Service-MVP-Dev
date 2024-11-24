from rec_system.method.recommendation import Recommender


class RecommendationService:
    def __init__(self):
        model_path = "src/rec_system/model/output/neumf_factor8neg4_Epoch4_HR1.0000_NDCG1.0000.model"
        config_path = "src/rec_system/model/output/config/config_epoch_4.pkl"
        self.recommender = Recommender(model_path, config_path)

    def recommend_for_new_item(self, item_data):
        return self.recommender.recommend_for_new_item(item_data)

    def recommend_for_new_creator(self, creator_data):
        return self.recommender.recommend_for_new_creator(creator_data)
