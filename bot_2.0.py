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

# preset save
from presets import save_Preset, load_Preset, exist_Preset, delete_Preset, show_Preset, Preset, prompt_preset_load, prompt_preset_delete, prompt_preset_save

def image_to_bytes(image):
    bio = BytesIO()
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')
    bio.seek(0)
    return bio

def process_prompt(prompt):
    err = 0
    if "!l" in prompt.lower(): 
        err, wanted_prompt = prompt_preset_load(prompt,"!l")
    else: 
        wanted_prompt=prompt
    wanted_prompt=" ".join(filter(lambda x:x[0]!='!', wanted_prompt.split())) #remove all keywords starting with !
    return wanted_prompt, err

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
    
def get_variations_markup():
    keyboard = [[InlineKeyboardButton("Variations", callback_data="VARIATIONS")]]
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
    wanted_prompt, err = process_prompt(update.message.text)
    if err:
        print("tried to load a nonexisting preset")
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        progress_msg = await update.message.reply_text("One or more presets do not exist. They will be ignored. \n(type !sl for the preset list)\n\nI keep generating...", reply_to_message_id=update.message.message_id)
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
    wanted_prompt, err = process_prompt(update.message.caption)
    if err:
        print("tried to load a nonexisting preset")
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        progress_msg = await update.message.reply_text("One or more presets do not exist. They will be ignored. \n(type !sl for the preset list)\n\nI keep generating...", reply_to_message_id=update.message.message_id)
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

async def save_preset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prompt=update.message.text
    
    if "!sl" in prompt.lower(): 
        message=[]
        for press in show_Preset(): 
            message.append(press.name+": ["+press.string+"] last modified: "+press.date+" used "+str(press.usages)+" times")
        message="\n".join(message)
        progress_msg = await update.message.reply_text(message, reply_to_message_id=update.message.message_id)
        return

    if "!sd" in prompt.lower(): 
        prompt_preset_delete(prompt, load_str="!sd")
        progress_msg = await update.message.reply_text("Preset deleted", reply_to_message_id=update.message.message_id)
        return

    if "!s!" in prompt.lower(): 
        err, name = prompt_preset_save(prompt, override=True)
    else:
        err, name = prompt_preset_save(prompt, override=False)
    if not err:
        progress_msg = await update.message.reply_text(name+" configuration saved sucessfuly", reply_to_message_id=update.message.message_id)
    else:
        progress_msg = await update.message.reply_text("Error: "+name+" not saved (it might already exist, in case you want to overwrite use !s!)", reply_to_message_id=update.message.message_id)
    
    
async def help_me(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("Help request ...")
    message =   ("!d to dream " 
                "\n!r + photo + caption to change it" 
                "\n!u to upscale a photo " 
                "\n - - - presets - - -" 
                "\n!sl to check the preset list" 
                "\n!s <<name>> to save a new preset"  
                "\n!s! <<name>> save and substitute a new preset" 
                "\n!sd <<name>> delete a preset"  
                "\n!l <<name>> in a prompt to load a preset")
    help_msg = await update.message.reply_text(message, reply_to_message_id=update.message.message_id)
    







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
            wanted_prompt, err = process_prompt(prompt)
            if err:
                print("tried to load a nonexisting preset")
                await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
                progress_msg = await update.message.reply_text("One or more presets do not exist. They will be ignored. \n(type !sl for the preset list)\n\nI keep generating...", reply_to_message_id=update.message.message_id)
            im, seed = generate_image(wanted_prompt, photo=photo)
        else:
            prompt = replied_message.text            
            wanted_prompt, err = process_prompt(prompt)
            if err:
                print("tried to load a nonexisting preset")
                await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
                progress_msg = await update.message.reply_text("One or more presets do not exist. They will be ignored. \n(type !sl for the preset list)\n\nI keep generating...", reply_to_message_id=update.message.message_id)
            im, seed = generate_image(wanted_prompt)
            
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'"{wanted_prompt}" (seed={seed})', reply_markup=get_try_again_markup(), reply_to_message_id=replied_message.message_id)
    
    elif query.data == "VARIATIONS":
        progress_msg = await query.message.reply_text("Generating image...", reply_to_message_id=replied_message.message_id)
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        prompt = replied_message.text if replied_message.text is not None else replied_message.caption
        wanted_prompt, err = process_prompt(prompt)
        if err:
            print("tried to load a nonexisting preset")
            await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
            progress_msg = await update.message.reply_text("One or more presets do not exist. They will be ignored. \n(type !sl for the preset list)\n\nI keep generating...", reply_to_message_id=update.message.message_id)
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
        await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'Size multiplied x4', reply_markup=get_try_again_markup(),reply_to_message_id=replied_message.message_id)
    elif query.data == "ENHANCE_ANIME":
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        progress_msg = await query.message.reply_text("Animization beammmmm!", reply_to_message_id=replied_message.message_id)
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        init_image = numpy.array(init_image)
        im = realesrgan_henance(input=init_image, outscale=4,model_name="RealESRGAN_x4plus_anime_6B",face_enhance=False)
        im = Image.fromarray(im)
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'Size multiplied x4',reply_markup=get_variations_markup(), reply_to_message_id=replied_message.message_id)
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
        im = realesrgan_henance(input=init_image, outscale=4,model_name="RealESRGAN_x4plus",face_enhance=True)
        im = Image.fromarray(im)
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'Size multiplied x4, faces fixed',reply_markup=get_variations_markup(), reply_to_message_id=replied_message.message_id)
    elif query.data == "ENHANCE_ANIME_FF":
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        progress_msg = await query.message.reply_text("Animization beammmmm! + face fixing", reply_to_message_id=replied_message.message_id)
        prompt = replied_message.text if replied_message.text is not None else replied_message.caption
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        init_image = numpy.array(init_image)
        im = realesrgan_henance(input=init_image, outscale=4,model_name="RealESRGAN_x4plus_anime_6B",face_enhance=True)
        im = Image.fromarray(im)
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_photo(replied_message.chat_id, image_to_bytes(im), caption=f'Size multiplied x4, faces fixed',reply_markup=get_variations_markup(), reply_to_message_id=replied_message.message_id)




# Here we have the actual app loop
app = ApplicationBuilder().token(bot_token).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & chatfilter & filters.Regex(re.compile(r'!d', re.IGNORECASE)), generate_and_send_photo))

app.add_handler(MessageHandler(filters.CaptionRegex(re.compile(r'!r', re.IGNORECASE)), generate_and_send_photo_from_photo))
app.add_handler(MessageHandler(filters.CaptionRegex(re.compile(r'!u', re.IGNORECASE)), rescale_photo ))

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & chatfilter & filters.Regex(re.compile(r'!s', re.IGNORECASE)), save_preset))

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & chatfilter & filters.Regex(re.compile(r'!h', re.IGNORECASE)), help_me))



app.add_handler(CallbackQueryHandler(button))
print("Ready")
app.run_polling()
