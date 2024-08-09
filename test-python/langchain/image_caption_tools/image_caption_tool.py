import torch
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from io import BytesIO
from langchain.tools import tool


@tool
def image_caption(img_url: str):
    """use this tool when given the url of an image that you would like to be
    described. It will return a simple caption describing the image."""

    hf_model = "Salesforce/blip-image-captioning-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained(hf_model)
    model = BlipForConditionalGeneration.from_pretrained(hf_model).to(device)

    # img_url = "https://ix-www.imgix.net/case-study/unsplash/woman-hat.jpg?ixlib=js-3.8.0&w=400&auto=compress%2Cformat&dpr=1&q=75"
    img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    img.show()

    inputs = processor(img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20)

    result = processor.decode(out[0], skip_special_tokens=True)
    print(result)
    return result

if __name__ == "__main__":
    img_url = "https://ix-www.imgix.net/case-study/unsplash/woman-hat.jpg?ixlib=js-3.8.0&w=400&auto=compress%2Cformat&dpr=1&q=75"
    image_caption(img_url)