from fastapi import APIRouter, HTTPException
from rec_system.schemas import ItemRecommendRequest,CreatorRecommendRequest,CreatorRecommendResponse,ItemRecommendResponse
from rec_system.service import RecommendationService

router = APIRouter()
recommendation_service = RecommendationService()


@router.post("/ai/recommend/item", response_model=CreatorRecommendResponse)
def recommend_item(data: ItemRecommendRequest):
    try:
        recommendations = recommendation_service.recommend_for_new_item(data.dict())
        return {
            "recommended_creators": [
                {
                    "creator_id": rec["creator_id"],
                    "channel_category": rec["channel_category"],
                    "channel_name": rec["channel_name"],
                    "subscribers": int(rec["subscribers"].replace(",", ""))  # 문자열을 정수로 변환
                }
                for rec in recommendations
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai/recommend/creator", response_model=ItemRecommendResponse)
def recommend_creator(data: CreatorRecommendRequest):
    try:
        recommendations = recommendation_service.recommend_for_new_creator(data.dict())
        return {
            "recommended_items": [
                {
                    "item_id": rec["item_id"],
                    "title": rec["title"],
                    "item_category": rec["item_category"],
                    "media_type": rec["media_type"],
                    "score": rec["score"],
                    "item_content": rec["item_content"]
                }
                for rec in recommendations
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))