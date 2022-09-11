# - - - - general import - - - -
import torch
from torch import autocast
from PIL import Image
import numpy
import random
import re
from io import BytesIO

# - - - - Stable Diffusion - - - - 
#from stable_diffusion_telegram import generate_image, preprocess
import stable_diffusion_telegram
from stable_diffusion_telegram import generate_image

# - - - - Enhance Image - - - - -
from inference_realesrgan_telegram import realesrgan_henance

# - - - - telegram bot - - - - 
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from telebot.credentials import chat_ids, bot_token, safety_checker, image_h, image_w


def image_to_bytes(image):
    bio = BytesIO()
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')
    bio.seek(0)
    return bio




# set up the chat filters
#nofilter
chatfilter = filters.TEXT
# yesfilter
#chatfilter = filters.Chat(chat_ids[0])
#chatfilter.chat_ids = chat_ids


# buttons 
def get_try_again_markup():
    keyboard = [[InlineKeyboardButton("Try again", callback_data="TRYAGAIN"), InlineKeyboardButton("Variations", callback_data="VARIATIONS")], [InlineKeyboardButton("Enahance", callback_data="ENHANCE"), InlineKeyboardButton("En+FixFaces", callback_data="ENHANCE_FF")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup

def get_enhance_markup():
    keyboard = [[InlineKeyboardButton("standard", callback_data="ENHANCE_STD"),InlineKeyboardButton("anime", callback_data="ENHANCE_ANIME")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup
    
def get_enhanceFF_markup():
    keyboard = [[InlineKeyboardButton("standard + faces", callback_data="ENHANCE_STD_FF"),InlineKeyboardButton("anime + faces", callback_data="ENHANCE_ANIME_FF")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup


      
async def generate_and_send_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    wanted_prompt=" ".join(filter(lambda x:x[0]!='!', update.message.text.split()))
    #wanted_prompt=re.sub('!dream', '', update.message.text, flags=re.IGNORECASE) #add this to remove !dream
    im, seed = generate_image(prompt= wanted_prompt)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.message.chat_id, image_to_bytes(im), caption=f'"{wanted_prompt}" (Seed: {seed})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)

async def generate_and_send_photo_from_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.caption is None:
        await update.message.reply_text("The photo must contain a text in the caption", reply_to_message_id=update.message.message_id)
        return
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    photo_file = await update.message.photo[-1].get_file()
    photo = await photo_file.download_as_bytearray()
    wanted_prompt=" ".join(filter(lambda x:x[0]!='!', update.message.text.split()))
    im, seed = generate_image(prompt=wanted_prompt, photo=photo)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.message.chat_id, image_to_bytes(im), caption=f'"{wanted_prompt}" (Seed: {seed})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)

async def rescale_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    progress_msg = await update.message.reply_text("rescaling...", reply_to_message_id=update.message.message_id)
    photo_file   = await update.message.photo[-1].get_file()
    photo        = await photo_file.download_as_bytearray()
    init_image = Image.open(BytesIO(photo)).convert("RGB")
    init_image = numpy.array(init_image)
    #im = upsampler_ESRGANer(photo=photo)
    #'Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | realesr-animevideov3'
    # realesrgan | bicubic
    im = realesrgan_henance(input=init_image, outscale=4,model_name="RealESRGAN_x4plus")
    im = Image.fromarray(im)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.message.chat_id, image_to_bytes(im), caption=f'Rescaled photo', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)





async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    replied_message = query.message.reply_to_message
    await query.answer()

    if query.data == "TRYAGAIN":
        progress_msg = await query.message.reply_text("Generating image...", reply_to_message_id=replied_message.message_id)
        if replied_message.photo is not None and len(replied_message.photo) > 0 and replied_message.caption is not None:
            photo_file = await replied_message.photo[-1].get_file()
            photo = await photo_file.download_as_bytearray()
            prompt = replied_message.caption
            wanted_prompt=" ".join(filter(lambda x:x[0]!='!', prompt.split()))
            im, seed = generate_image(wanted_prompt, photo=photo)
        else:
            prompt = replied_message.text
            wanted_prompt=" ".join(filter(lambda x:x[0]!='!', prompt.split()))
            im, seed = generate_image(wanted_prompt)
            
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'"{wanted_prompt}" (seed={seed})', reply_markup=get_try_again_markup(), reply_to_message_id=replied_message.message_id)
    
    elif query.data == "VARIATIONS":
        progress_msg = await query.message.reply_text("Generating image...", reply_to_message_id=replied_message.message_id)
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        prompt = replied_message.text if replied_message.text is not None else replied_message.caption
        wanted_prompt=" ".join(filter(lambda x:x[0]!='!', prompt.split()))
        im, seed = generate_image(wanted_prompt, photo=photo)
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'"{wanted_prompt}" (seed={seed})', reply_markup=get_try_again_markup(), reply_to_message_id=replied_message.message_id) 
        
        
        
    elif query.data == "ENHANCE":
        #await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        init_image = numpy.array(init_image)
        im = Image.fromarray(init_image)
        await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'Model to be used:', reply_markup=get_enhance_markup(), reply_to_message_id=replied_message.message_id)
        #await context.bot.send_message(replied_message.chat_id, text=f"Model to be used:", reply_markup=get_enhance_markup(), reply_to_message_id=replied_message.message_id)
    elif query.data == "ENHANCE_STD":
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        progress_msg = await query.message.reply_text("Enhancing image...", reply_to_message_id=replied_message.message_id)
        prompt = replied_message.text if replied_message.text is not None else replied_message.caption
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        init_image = numpy.array(init_image)
        im = realesrgan_henance(input=init_image, outscale=4,model_name="RealESRGAN_x4plus",face_enhance=False)
        im = Image.fromarray(im)
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'Size multiplied x4', reply_to_message_id=replied_message.message_id)
    elif query.data == "ENHANCE_ANIME":
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        progress_msg = await query.message.reply_text("Animization beammmmm!", reply_to_message_id=replied_message.message_id)
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        init_image = numpy.array(init_image)
        im = realesrgan_henance(input=init_image, outscale=4,model_name="RealESRGAN_x4plus_anime_6B",face_enhance=False)
        im = Image.fromarray(im)
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'Size multiplied x4', reply_to_message_id=replied_message.message_id)
    elif query.data == "ENHANCE_FF":
        #await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        init_image = numpy.array(init_image)
        im = Image.fromarray(init_image)
        await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'Model to be used:', reply_markup=get_enhanceFF_markup(), reply_to_message_id=replied_message.message_id)
        #await context.bot.send_message(replied_message.chat_id, text=f"Model to be used:", reply_markup=get_enhance_markup(), reply_to_message_id=replied_message.message_id)
    elif query.data == "ENHANCE_STD_FF":
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        progress_msg = await query.message.reply_text("Enhancing image... + face fixing", reply_to_message_id=replied_message.message_id)
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        init_image = numpy.array(init_image)
        im = realesrgan_henance(input=init_image, outscale=4,model_name="RealESRGAN_x4plus",face_enhance=False)
        im = Image.fromarray(im)
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'Size multiplied x4, faces fixed', reply_to_message_id=replied_message.message_id)
    elif query.data == "ENHANCE_ANIME_FF":
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        progress_msg = await query.message.reply_text("Animization beammmmm! + face fixing", reply_to_message_id=replied_message.message_id)
        prompt = replied_message.text if replied_message.text is not None else replied_message.caption
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        init_image = numpy.array(init_image)
        im = realesrgan_henance(input=init_image, outscale=4,model_name="RealESRGAN_x4plus_anime_6B",face_enhance=False)
        im = Image.fromarray(im)
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'Size multiplied x4, faces fixed', reply_to_message_id=replied_message.message_id)




# Here we have the actual app loop
app = ApplicationBuilder().token(bot_token).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & chatfilter & filters.Regex(re.compile(r'!dream', re.IGNORECASE)), generate_and_send_photo))
app.add_handler(MessageHandler(filters.CaptionRegex(re.compile(r'!redream', re.IGNORECASE)), generate_and_send_photo_from_photo))
#app.add_handler(MessageHandler(filters.CaptionRegex(re.compile(r'!face', re.IGNORECASE)), generate_and_send_faces_from_photo))
app.add_handler(MessageHandler(filters.CaptionRegex(re.compile(r'!upscale', re.IGNORECASE)), rescale_photo ))



app.add_handler(CallbackQueryHandler(button))
print("Ready")
app.run_polling()
