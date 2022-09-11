from telegram import Update
from telegram.ext import ContextTypes
import re
import random


class OrsoClass:
    def __init__(self, config):
        self.config = config
        self.gufi_list = ["AgACAgQAAxkBAAEXnaBjDlsg-hKNljrwCec6LMrbbbq0PAACJLsxGyPLeVAL_2pFs8Hw9gEAAwIAA3kAAykE",
                          "AgACAgQAAxkBAAEXnaFjDlsgu0qQgqmmge3tJX-6hRpsogACI7sxGyPLeVDMGt8XNGxxkwEAAwIAA3kAAykE",
                          "AgACAgQAAxkBAAEXnaJjDlsgp_qM7Aa-thdLPmxtg66WQwACIbsxGyPLeVCaXLNibYmM6gEAAwIAA3gAAykE",
                          "AgACAgQAAxkBAAEXnaNjDlsguzwDrR6hB72CBuu6FRKDOQACILsxGyPLeVCjgsyFT3yiIAEAAwIAA3gAAykE",
                          "AgACAgQAAxkBAAEXnaRjDlsgAAHzxpw8br86idhEECMramwAAiK7MRsjy3lQRyxMTTMyasUBAAMCAAN5AAMpBA"]

    def reload_config(self, config):
        self.config = config

    async def gufa(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_photo(random.choice(self.gufi_list))

    async def faccia(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("La tua faccia è" + re.split('sei', update.message.text, flags=re.IGNORECASE)[-1])

    async def rosa(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("La merda rosa")

    async def sette(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(random.choice(["Sette", "Circa sette", "7", "@settesette77777"]))

    async def si(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(random.choice(["Sì", "Esatto", "Sì", "Sì", "Renzi."]))

    async def sticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_sticker(random.choice(self.config['ORSOBOT PARAMS']['sticker_list']))
