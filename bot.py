#!/usr/bin/env python
import argparse

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from image_to_image import StableDiffusionImg2ImgPipeline, preprocess
from PIL import Image

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes, MessageHandler, filters, CommandHandler
from telebot.orsobot import OrsoClass
from telebot.aaarghparser import init_parser
from io import BytesIO
import random
import logging
import configparser
import json
import re
import asyncio


# INITIAL setup of config and logger
config_file = 'config.ini'
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read(config_file)
parser = init_parser()

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# disable safety checker if wanted
def dummy_checker(images, **kwargs): return images, False


# Image gen Funcions (TODO: move to another file?)
def image_to_bytes(image):
    bio = BytesIO()
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')
    bio.seek(0)
    return bio


def get_try_again_markup():
    keyboard = [[InlineKeyboardButton("Try again", callback_data="TRYAGAIN"),
                 InlineKeyboardButton("Variations", callback_data="VARIATIONS")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup


def generate_image(args, photo=None):
    seed = args.seed if args.seed is not None else torch.seed()
    logger.debug(args)
    generator = torch.cuda.manual_seed_all(seed)
    prompt = " ".join(args.dream)
    prompt = 'Owlbear ' + prompt if random.randint(0, 100) < args.owlbearify else prompt

    image_h = args.image_h * 1.25**args.portrait * 1.25**-args.landscape
    image_w = args.image_w * 1.25**-args.portrait * 1.25**args.landscape

    # approximate h and w to the closes multiple of 64
    image_h = 64 * round(image_h / 64)
    image_w = 64 * round(image_w / 64)

    if photo is not None:
        pipe.to("cpu")
        img2imgPipe.to("cuda")
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        init_image = init_image.resize((image_w, image_h))
        init_image = preprocess(init_image)
        with autocast("cuda"):
            image = img2imgPipe(prompt=[prompt], init_image=init_image,
                                generator=generator,
                                strength=float(args.denoising_strength),
                                guidance_scale=float(args.guidance_scale),
                                num_inference_steps=args.steps)["sample"][0]
    else:
        pipe.to("cuda")
        img2imgPipe.to("cpu")
        with autocast("cuda"):
            image = pipe(prompt=[prompt],
                         height=image_h,
                         width=image_w,
                         generator=generator,
                         strength=float(args.denoising_strength),
                         guidance_scale=float(args.guidance_scale),
                         num_inference_steps=int(args.steps))["sample"][0]
    return image, seed


async def generate_and_send_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    try:
        parser.set_defaults(**dict(config['GLOBALS']))
        parser.set_defaults(**dict(config['ORSOBOT PARAMS']))
        parser.set_defaults(**dict(config['TXT2IMG PARAMS']))
        args = parser.parse_args(update.message.text.split())
        progress_msg = await update.message.reply_text("Generating image...",
                                                       reply_to_message_id=update.message.message_id)

        im, seed = generate_image(args=args)
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(update.message.chat_id, image_to_bytes(im), caption=f'"{update.message.text}" (Seed: {seed})',
                                     reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)
    except (argparse.ArgumentError, argparse.ArgumentTypeError, ValueError) as exc:
        await update.message.reply_text(str(exc) + '\n\n' + parser.format_help())


async def generate_and_send_photo_from_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.caption is None:
        await update.message.reply_text("The photo must contain a text in the caption", reply_to_message_id=update.message.message_id)
        return

    photo_file = await update.message.photo[-1].get_file()
    photo = await photo_file.download_as_bytearray()
    try:
        progress_msg = await update.message.reply_text("Generating image...",
                                                       reply_to_message_id=update.message.message_id)
        parser.set_defaults(**dict(config['GLOBALS']))
        parser.set_defaults(**dict(config['ORSOBOT PARAMS']))
        parser.set_defaults(**dict(config['IMG2IMG PARAMS']))
        args = parser.parse_args(update.message.caption.split())
        im, seed = generate_image(args=args, photo=photo)
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(update.message.chat_id, image_to_bytes(im),
                                     caption=f'"{update.message.caption}" (Seed: {seed})',
                                     reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)
    except (argparse.ArgumentError, argparse.ArgumentTypeError, ValueError) as exc:
        await update.message.reply_text(str(exc) + '\n\n' + parser.format_help())


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    replied_message = query.message.reply_to_message

    await query.answer()
    progress_msg = await query.message.reply_text("Generating image...", reply_to_message_id=replied_message.message_id)

    if query.data == "TRYAGAIN":
        if replied_message.photo is not None and len(replied_message.photo) > 0 and replied_message.caption is not None:
            photo_file = await replied_message.photo[-1].get_file()
            photo = await photo_file.download_as_bytearray()
            parser.set_defaults(**dict(config['GLOBALS']))
            parser.set_defaults(**dict(config['ORSOBOT PARAMS']))
            parser.set_defaults(**dict(config['IMG2IMG PARAMS']))
            msg = replied_message.caption
            args = parser.parse_args(msg.split())
            im, seed = generate_image(args=args, photo=photo)
        else:
            parser.set_defaults(**dict(config['GLOBALS']))
            parser.set_defaults(**dict(config['IMG2IMG PARAMS']))
            msg = replied_message.text
            args = parser.parse_args(msg.split())
            im, seed = generate_image(args=args)
    elif query.data == "VARIATIONS":
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        parser.set_defaults(**dict(config['GLOBALS']))
        parser.set_defaults(**dict(config['ORSOBOT PARAMS']))
        parser.set_defaults(**dict(config['IMG2IMG PARAMS']))
        msg = replied_message.text if replied_message.text is not None else replied_message.caption
        args = parser.parse_args(msg.split())
        im, seed = generate_image(args=args, photo=photo)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'"{msg}" (Seed: {seed})',
                                 reply_markup=get_try_again_markup(), reply_to_message_id=replied_message.message_id)

# LOAD THE PIPES
# load the text2img pipeline
revision = "fp16" if config.getboolean('GLOBALS', 'low_memory') else None
torch_dtype = torch.float16 if config.getboolean('GLOBALS', 'low_memory') else None

pipe = StableDiffusionPipeline.from_pretrained(config['GLOBALS']['SD_checkpoint'], revision=revision,
                                               torch_dtype=torch_dtype, use_auth_token=True)
pipe = pipe.to("cpu")

# load the img2img pipeline
img2imgPipe = StableDiffusionImg2ImgPipeline.from_pretrained(config['GLOBALS']['SD_checkpoint'], revision=revision,
                                                             torch_dtype=torch_dtype, use_auth_token=True)
img2imgPipe = img2imgPipe.to("cpu")


# Here we have the actual app loop
def main():
    global app
    config.read(config_file)
    if not config.getboolean('GLOBALS', 'safety_check'): #TODO: the filter cannot be reenabled once disabled this way
        pipe.safety_checker = dummy_checker
        img2imgPipe.safety_checker = dummy_checker
    # set up the chat filters
    try:
        chat_ids = json.loads(config['GLOBALS']['allowed_chats'])
        chatfilter = filters.Chat(chat_ids[0])
        chatfilter.chat_ids = chat_ids
    except KeyError:
        chatfilter = filters.TEXT
    try:
        app = ApplicationBuilder().token(config['GLOBALS']['bot_token']).build()
    except:
        pass
    print('blub')
    try:
        for handler in app.handlers[0]:
            app.remove_handler(handler)
    except KeyError:
        logger.info("No handlers to remove?")
    app.add_handler(CommandHandler("reload", boh, filters=chatfilter))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & chatfilter
                                   & filters.Regex(re.compile(r'--dream|-d', re.IGNORECASE)), generate_and_send_photo))
    app.add_handler(MessageHandler(filters.PHOTO & chatfilter & filters.Regex(re.compile(r'--dream|-d', re.IGNORECASE)),
                                   generate_and_send_photo_from_photo))
    if config.getboolean('ORSOBOT PARAMS', 'enable_owlbear', fallback=False):
        owlbear = OrsoClass(config)
        namelist = json.loads(config['GLOBALS']['bot_names'])
        namestring = r"".join(s + "|" for s in namelist)
        app.add_handler(MessageHandler(filters.Regex(re.compile(namestring[:-1], re.IGNORECASE)) & chatfilter, owlbear.gufa))
        app.add_handler(MessageHandler(filters.Regex(re.compile(r'sei', re.IGNORECASE)) & chatfilter, owlbear.faccia))
        app.add_handler(MessageHandler((filters.Regex(re.compile(r'sticker', re.IGNORECASE)) |
                                        filters.Regex(re.compile(r'landreoli', re.IGNORECASE))) & chatfilter, owlbear.sticker))
        app.add_handler(MessageHandler(filters.Regex(re.compile(r'cosa', re.IGNORECASE)) & chatfilter, owlbear.rosa))
        app.add_handler(MessageHandler(filters.Regex(re.compile(r' o ', re.IGNORECASE)) & chatfilter, owlbear.si))
        app.add_handler(MessageHandler(filters.Regex(re.compile(r'numero ', re.IGNORECASE)) & chatfilter, owlbear.sette))
    app.add_handler(CallbackQueryHandler(button))

    app.run_polling()


# Function for the reload TODO: this does not really work del tutto ma vabbè non ci capisco una mazza
async def boh(a, b):
    global app
    loop = asyncio.get_event_loop()
    try:
        if app.updater.running:  # type: ignore[union-attr]
            loop.run_until_complete(app.updater.stop())  # type: ignore[union-attr]
        if app.running:
            loop.run_until_complete(app.stop())
        # loop.run_until_complete(app.shutdown())
        # if app.post_shutdown:
        #    loop.run_until_complete(app.post_shutdown(app))
    finally:
        # loop.stop()
        # time.sleep(2)
        # loop.close()
        main()


# Launch everything
if __name__ == "__main__":
    main()
