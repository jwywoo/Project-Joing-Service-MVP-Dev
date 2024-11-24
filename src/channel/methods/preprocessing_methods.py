import base64

from PIL import Image
from io import BytesIO


def response_preprocessing(playlist_response):
    text_info = []
    thumbnail_urls = []
    for item in playlist_response["items"]:
        snippet = item["snippet"]
        # title
        video_title = snippet["title"]
        # description
        video_desc = snippet['description']
        # thumbnails - urls
        video_thumbnail_url = snippet['thumbnails']['standard']['url']
        text_info.append({
            "title": video_title,
            "description": video_desc,
        })
        thumbnail_urls.append(video_thumbnail_url)
    return text_info, thumbnail_urls


def image_preprocessing(image_response):
    img_in_bytes = []
    for i in range(len(image_response)):
        img_in_bytes.append(Image.open(BytesIO(image_response[i].content)))

    img_width = img_in_bytes[0].size[0]
    img_height = img_in_bytes[0].size[1]

    combined_width = img_width*2
    combined_height = img_height*2

    combined_image = Image.new('RGB', (combined_width, combined_height))

    combined_image.paste(img_in_bytes[0], (0, 0))
    combined_image.paste(img_in_bytes[1], (img_width, 0))
    combined_image.paste(img_in_bytes[2], (0, img_height))
    combined_image.paste(img_in_bytes[3], (img_width, img_height))

    buffer = BytesIO()
    combined_image.save(buffer, format="JPEG")
    buffer.seek(0)
    combined_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return combined_base64
