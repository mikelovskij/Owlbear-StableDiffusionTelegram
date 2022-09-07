import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from image_to_image import StableDiffusionImg2ImgPipeline, preprocess
from PIL import Image

import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from telebot.credentials import chat_ids, bot_token, safety_checker, image_h, image_w
from telebot.orsobot_functions import gufa, faccia, rosa, sette, si, sticker
from io import BytesIO
import random

import re


# should load these from a cfg file (reload every time we have a new run also)
NUM_INFERENCE_STEPS = int(os.getenv('NUM_INFERENCE_STEPS', '75'))
STRENTH = float(os.getenv('STRENTH', '0.75'))
GUIDANCE_SCALE = float(os.getenv('GUIDANCE_SCALE', '9'))

# set up the chat filters
chatfilter = filters.Chat(chat_ids[0])
chatfilter.chat_ids = chat_ids


# load the text2img pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True)
pipe = pipe.to("cpu")

# load the img2img pipeline
img2imgPipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",revision="fp16", torch_dtype=torch.float16, use_auth_token=True)
img2imgPipe = img2imgPipe.to("cpu")


# disable safety checker if wanted
def dummy_checker(images, **kwargs): return images, False


if not safety_checker:
    pipe.safety_checker = dummy_checker
    img2imgPipe.safety_checker = dummy_checker


def image_to_bytes(image):
    bio = BytesIO()
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')
    bio.seek(0)
    return bio


def get_try_again_markup():
    keyboard = [[InlineKeyboardButton("Try again", callback_data="TRYAGAIN"), InlineKeyboardButton("Variations", callback_data="VARIATIONS")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup


def generate_image(prompt, seed=None, height=image_h, width=image_w, num_inference_steps=NUM_INFERENCE_STEPS, strength=STRENTH, guidance_scale=GUIDANCE_SCALE, photo=None):
    seed = seed if seed is not None else torch.seed()
    generator = torch.cuda.manual_seed_all(seed)
    prompt = 'Owlbear ' + prompt.replace("!dream ", "") if random.randint(0, 5) == 0 else prompt.replace("!dream ", "")
    if photo is not None:
        pipe.to("cpu")
        img2imgPipe.to("cuda")
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        init_image = init_image.resize((height, width))
        init_image = preprocess(init_image)
        with autocast("cuda"):
            image = img2imgPipe(prompt=[prompt], init_image=init_image,
                                    generator=generator,
                                    strength=strength,
                                    guidance_scale=guidance_scale,
                                    num_inference_steps=num_inference_steps)["sample"][0]
    else:
        pipe.to("cuda")
        img2imgPipe.to("cpu")
        with autocast("cuda"):
            image = pipe(prompt=[prompt],
                                    generator=generator,
                                    strength=strength,
                                    height=height,
                                    width=width,
                                    guidance_scale=guidance_scale,
                                    num_inference_steps=num_inference_steps)["sample"][0]
    return image, seed


async def generate_and_send_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    im, seed = generate_image(prompt= update.message.text)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.message.chat_id, image_to_bytes(im), caption=f'"{update.message.text}" (Seed: {seed})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)


async def generate_and_send_photo_from_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.caption is None:
        await update.message.reply_text("The photo must contain a text in the caption", reply_to_message_id=update.message.message_id)
        return
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    photo_file = await update.message.photo[-1].get_file()
    photo = await photo_file.download_as_bytearray()
    im, seed = generate_image(prompt=update.message.caption, photo=photo)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.message.chat_id, image_to_bytes(im), caption=f'"{update.message.caption}" (Seed: {seed})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    replied_message = query.message.reply_to_message

    await query.answer()
    progress_msg = await query.message.reply_text("Generating image...", reply_to_message_id=replied_message.message_id)

    if query.data == "TRYAGAIN":
        if replied_message.photo is not None and len(replied_message.photo) > 0 and replied_message.caption is not None:
            photo_file = await replied_message.photo[-1].get_file()
            photo = await photo_file.download_as_bytearray()
            prompt = replied_message.caption
            im, seed = generate_image(prompt, photo=photo)
        else:
            prompt = replied_message.text
            im, seed = generate_image(prompt)
    elif query.data == "VARIATIONS":
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        prompt = replied_message.text if replied_message.text is not None else replied_message.caption
        im, seed = generate_image(prompt, photo=photo)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'"{prompt}" (Seed: {seed})', reply_markup=get_try_again_markup(), reply_to_message_id=replied_message.message_id)


# Here we have the actual app loop
app = ApplicationBuilder().token(bot_token).build()

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & chatfilter & filters.Regex(re.compile(r'!dream', re.IGNORECASE)), generate_and_send_photo))
app.add_handler(MessageHandler(filters.PHOTO & chatfilter, generate_and_send_photo_from_photo))
app.add_handler(MessageHandler((filters.Regex(re.compile(r'orso', re.IGNORECASE)) |
                                filters.Regex(re.compile(r'gufo', re.IGNORECASE))) & chatfilter, gufa))
app.add_handler(MessageHandler(filters.Regex(re.compile(r'sei', re.IGNORECASE)) & chatfilter, faccia))
app.add_handler(MessageHandler((filters.Regex(re.compile(r'sticker', re.IGNORECASE)) |
                                filters.Regex(re.compile(r'landreoli', re.IGNORECASE))) & chatfilter, sticker))
app.add_handler(MessageHandler(filters.Regex(re.compile(r'cosa', re.IGNORECASE)) & chatfilter, rosa))
app.add_handler(MessageHandler(filters.Regex(re.compile(r' o ', re.IGNORECASE)) & chatfilter, si))
app.add_handler(MessageHandler(filters.Regex(re.compile(r'numero ', re.IGNORECASE)) & chatfilter, sette))
app.add_handler(CallbackQueryHandler(button))

app.run_polling()
