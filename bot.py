import json
import os
import random
import torch
import tweepy

from datetime import datetime
from diffusers import StableDiffusionPipeline
from PIL import Image, PngImagePlugin


with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    cfg = json.load(f)

HF_TOKEN = cfg["HF_TOKEN"]
KEY = cfg["KEY"]
SECRET = cfg["SECRET"]
ATOKEN = cfg["ATOKEN"]
ASECRET = cfg["ASECRET"]

auth = tweepy.OAuthHandler(KEY, SECRET)
auth.set_access_token(ATOKEN, ASECRET)
api = tweepy.API(auth)

pipe = StableDiffusionPipeline.from_pretrained("hakurei/waifu-diffusion",
                                               use_auth_token=HF_TOKEN)


# disable safety checker
def null_safety(images, **kwargs):
    return images, False


pipe.safety_checker = null_safety

while True:
    seed = random.randint(0, 10000)
    generator = torch.Generator("cpu").manual_seed(seed)

    colors = ["black", "blonde", "silver", "white", "red", "blue", "brown", "pink"]
    hair_color = random.choice(colors)
    eye_color = random.choice(colors)
    place = random.choice(["indoors", "outdoors", "classroom", "bedroom", "explosion"])
    body = random.choice(["upper body", "cowboy shot", "full body", "wide shot"])
    clothes = random.choice(["serafuku", "school uniform", "sundress", "maid", "gloves"])
    hair_style = random.choice(["long hair", "short hair", "braid", "twintails", "ponytail"])
    posture = random.choice(["sitting", "standing", "lying", "kneeling"])

    prompt = f"{body}, 1girl, solo, {hair_color} hair, {hair_style}, {eye_color} eyes, {place}, {clothes}, {posture}, looking at viewer, safe"
    negative = "explicit, questionable, nsfw, pussy, head out of frame, bad anatomy, bad hands, lowres, blurry, cropped, jpeg artifacts, low quality, text, signature, chibi"

    steps = 50
    image = pipe(prompt, generator=generator, num_inference_steps=steps)["sample"][0]

    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{date}_{seed}.png"
    image.save(filename)

    # Embed information in PNG
    info = PngImagePlugin.PngInfo()
    info.add_text("text", f"prompt: \"{prompt}\", negative: \"{negative}\", seed: {seed}")
    img = Image.open(filename)
    img.save(filename, "PNG", pnginfo=info)

    try:
        api.update_status_with_media(status=prompt, filename=filename)
    except Exception as e:
        print(e)
