import PIL
import requests

from PIL import Image
from io import BytesIO

def response_preprocessing(playlist_response):
    videos = []
    for item in playlist_response["items"]:
        snippet = item["snippet"]
        # title
        video_title = snippet["title"]
        # description
        video_desc = snippet['description']
        # thumbnails - urls
        video_thumbnail_url = snippet['thumbnails']['standard']['url']
        videos.append({
            "title": video_title,
            "description": video_desc,
            "thumbnail_url": video_thumbnail_url
        })
    return videos

def image_preprocessing(image_response):
    img_in_bytes = []
    for i in range(len(image_response)):
        img_in_bytes.append(Image.open(BytesIO(image_response[i].content)))
    
    img_width = img_in_bytes[0].size[0]
    img_height = img_in_bytes[0].size[1]
    
    combined_width = img_width*2
    combined_height = img_height*2
    
    combined_image = Image.new*('RGB', (combined_width, combined_height))
    
    combined_image.paste(img_in_bytes[0],(0,0))
    combined_image.paste(img_in_bytes[1],(img_width, 0))
    combined_image.paste(img_in_bytes[2],(0, img_height))
    combined_image.paste(img_in_bytes[3],(img_width, img_height))
    
    combined_image.show()
    return combined_image
    
    
    