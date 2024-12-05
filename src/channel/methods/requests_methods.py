import requests
import googleapiclient.discovery

from fastapi import HTTPException


def youtube_data_api_request(api_key):
    youtube_data_api = googleapiclient.discovery.build(
        'youtube', 'v3', developerKey=api_key)
    return youtube_data_api


def youtube_channel_request(youtube_data_api, channel_id):
    channel_response = youtube_data_api.channels().list(
        part='contentDetails',
        id=channel_id
    ).execute()
    if (channel_response['pageInfo']['totalResults'] == 0):
        raise HTTPException(
            status_code=400,
            detail={
                "유효하지 않은 채널아이디입니다."
            }
        )
    return channel_response


def playlist_request(youtube_data_api, youtube_channel):
    uploads_playlist_id = youtube_channel["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    playlist_response = youtube_data_api.playlistItems().list(
        part='snippet',
        playlistId=uploads_playlist_id,
        maxResults=4,
        pageToken=None
    ).execute()
    
    if len(playlist_response) < 4:
        raise HTTPException(
            status_code=400,
            detail={
                "영상의 수가 부족하여 더이상의 평가가 불가능합니다."
            }
        )
    return playlist_response


def image_request(image_urls):
    image_responses = []
    for i in range(len(image_urls)):
        image_responses.append(requests.get(image_urls[i]))
    return image_responses
