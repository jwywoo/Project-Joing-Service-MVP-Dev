from pydantic import BaseModel


class CreatorRecommendRequest(BaseModel):
    user_id: int
    channel_name: str
    channel_category: str
    subscribers: int
    # additional_features: Optional[List[str]] = None
    # media_type: Optional[str] = None
    # max_views: Optional[int] = None
    # min_views: Optional[int] = None
    # comments: Optional[int] = None


class ItemRecommendRequest(BaseModel):
    item_id: int
    title: str
    item_category: str
    media_type: str
    score: int


class CreatorRecommendResponse(BaseModel):
    user_id: int
    channel_name: str
    channel_category: str
    subscribers: int


class ItemRecommendResponse(BaseModel):
    item_id: int
    title: str
    item_category: str
    media_type: str
    score: int
