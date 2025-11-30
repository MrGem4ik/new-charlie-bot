# RENDER DEPLOYMENT - PYTHON ANYWHERE REPLACEMENT
import os
import logging
import random
import google.generativeai as genai
from datetime import datetime, date, timedelta
from collections import defaultdict
import telebot
from telebot import types
from gtts import gTTS
import time
import re
import requests
import io
import os
import logging
import random
import google.generativeai as genai
from datetime import datetime, date, timedelta
from collections import defaultdict
import telebot
from telebot import types
from gtts import gTTS
import time
import re
import requests
import io
import wave
from threading import Thread
import subprocess
import sys
import sqlite3
import json

# РРјРїРѕСЂС‚РёСЂСѓРµРј РєРѕРЅС„РёРі
from config import TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, CRYPTO_BOT_TOKEN
# РќР°СЃС‚СЂРѕР№РєР° Р»РѕРіРёСЂРѕРІР°РЅРёСЏ
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ Р±РѕС‚Р°
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Р“Р»РѕР±Р°Р»СЊРЅС‹Рµ РїРµСЂРµРјРµРЅРЅС‹Рµ РґР»СЏ С…СЂР°РЅРµРЅРёСЏ РёСЃС‚РѕСЂРёРё РґРёР°Р»РѕРіРѕРІ Рё СЏР·С‹РєРѕРІС‹С… РЅР°СЃС‚СЂРѕРµРє
user_conversations = defaultdict(lambda: [])
user_languages = defaultdict(lambda: 'ru')
user_voice_enabled = defaultdict(lambda: True)
chat_voice_support = defaultdict(lambda: True)


# Р‘Р°Р·Р° РґР°РЅРЅС‹С… РїРѕР»СЊР·РѕРІР°С‚РµР»РµР№
class UserDatabase:
    def __init__(self, db_path='users.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ Р±Р°Р·С‹ РґР°РЅРЅС‹С…"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                is_premium BOOLEAN DEFAULT FALSE,
                premium_until DATE,
                stars INTEGER DEFAULT 0,
                voice_uses_today INTEGER DEFAULT 0,
                last_voice_date DATE,
                explicit_mode BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # РџСЂРѕРІРµСЂСЏРµРј СЃСѓС‰РµСЃС‚РІРѕРІР°РЅРёРµ СЃС‚РѕР»Р±С†Р° explicit_mode Рё РґРѕР±Р°РІР»СЏРµРј РµСЃР»Рё РЅСѓР¶РЅРѕ
        try:
            cursor.execute("SELECT explicit_mode FROM users LIMIT 1")
        except sqlite3.OperationalError:
            logger.info("Р”РѕР±Р°РІР»СЏРµРј СЃС‚РѕР»Р±РµС† explicit_mode РІ С‚Р°Р±Р»РёС†Сѓ users")
            cursor.execute('ALTER TABLE users ADD COLUMN explicit_mode BOOLEAN DEFAULT FALSE')

        conn.commit()
        conn.close()
        logger.info("вњ… Р‘Р°Р·Р° РґР°РЅРЅС‹С… РїРѕР»СЊР·РѕРІР°С‚РµР»РµР№ РёРЅРёС†РёР°Р»РёР·РёСЂРѕРІР°РЅР°")

    def get_user(self, user_id):
        """РџРѕР»СѓС‡РёС‚СЊ РґР°РЅРЅС‹Рµ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_id, username, first_name, last_name, is_premium, 
                   premium_until, stars, voice_uses_today, last_voice_date,
                   explicit_mode, created_at
            FROM users WHERE user_id = ?
        ''', (user_id,))
        user = cursor.fetchone()
        conn.close()

        if user:
            return {
                'user_id': user[0],
                'username': user[1],
                'first_name': user[2],
                'last_name': user[3],
                'is_premium': bool(user[4]),
                'premium_until': user[5],
                'stars': user[6],
                'voice_uses_today': user[7],
                'last_voice_date': user[8],
                'explicit_mode': bool(user[9]),
                'created_at': user[10]
            }
        return None

    def create_user(self, user_id, username, first_name, last_name):
        """РЎРѕР·РґР°С‚СЊ РЅРѕРІРѕРіРѕ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # РџСЂРѕРІРµСЂСЏРµРј СЃСѓС‰РµСЃС‚РІРѕРІР°РЅРёРµ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ
        cursor.execute('SELECT 1 FROM users WHERE user_id = ?', (user_id,))
        exists = cursor.fetchone()

        if not exists:
            cursor.execute('''
                INSERT INTO users 
                (user_id, username, first_name, last_name, stars, explicit_mode)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, username, first_name, last_name, 0, False))
        else:
            # РћР±РЅРѕРІР»СЏРµРј РґР°РЅРЅС‹Рµ СЃСѓС‰РµСЃС‚РІСѓСЋС‰РµРіРѕ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ
            cursor.execute('''
                UPDATE users 
                SET username = ?, first_name = ?, last_name = ?
                WHERE user_id = ?
            ''', (username, first_name, last_name, user_id))

        conn.commit()
        conn.close()

    def update_stars(self, user_id, stars):
        """РћР±РЅРѕРІРёС‚СЊ РєРѕР»РёС‡РµСЃС‚РІРѕ Р·РІРµР·Рґ"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET stars = ? WHERE user_id = ?
        ''', (stars, user_id))
        conn.commit()
        conn.close()

    def activate_premium(self, user_id, days=7):
        """РђРєС‚РёРІРёСЂРѕРІР°С‚СЊ РїСЂРµРјРёСѓРј РїРѕРґРїРёСЃРєСѓ"""
        premium_until = datetime.now() + timedelta(days=days)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users 
            SET is_premium = TRUE, premium_until = ?
            WHERE user_id = ?
        ''', (premium_until.strftime('%Y-%m-%d'), user_id))
        conn.commit()
        conn.close()

    def toggle_explicit_mode(self, user_id):
        """РџРµСЂРµРєР»СЋС‡РёС‚СЊ СЂРµР¶РёРј РѕС‚РєСЂРѕРІРµРЅРЅС‹С… С‚РµРј"""
        user = self.get_user(user_id)
        if user and user['is_premium']:
            new_mode = not user['explicit_mode']
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET explicit_mode = ? WHERE user_id = ?
            ''', (new_mode, user_id))
            conn.commit()
            conn.close()
            return new_mode
        return False

    def add_stars(self, user_id, amount):
        """Р”РѕР±Р°РІРёС‚СЊ Р·РІРµР·РґС‹ РїРѕР»СЊР·РѕРІР°С‚РµР»СЋ"""
        user = self.get_user(user_id)
        if user:
            new_stars = user['stars'] + amount
            self.update_stars(user_id, new_stars)
            return new_stars
        return 0

    def can_use_voice(self, user_id):
        """РџСЂРѕРІРµСЂРёС‚СЊ, РјРѕР¶РµС‚ Р»Рё РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ РІРѕР№СЃС‹ СЃРµРіРѕРґРЅСЏ"""
        user = self.get_user(user_id)
        if not user:
            return True

        today = date.today()
        last_date = user['last_voice_date']

        # Р•СЃР»Рё РїРѕСЃР»РµРґРЅРµРµ РёСЃРїРѕР»СЊР·РѕРІР°РЅРёРµ Р±С‹Р»Рѕ РЅРµ СЃРµРіРѕРґРЅСЏ, СЃР±СЂР°СЃС‹РІР°РµРј СЃС‡РµС‚С‡РёРє
        if last_date != str(today):
            self.reset_voice_counter(user_id)
            return True

        # РџСЂРѕРІРµСЂСЏРµРј Р»РёРјРёС‚
        if user['is_premium']:
            return True
        else:
            return user['voice_uses_today'] < 3

    def increment_voice_use(self, user_id):
        """РЈРІРµР»РёС‡РёС‚СЊ СЃС‡РµС‚С‡РёРє РёСЃРїРѕР»СЊР·РѕРІР°РЅРёР№ РІРѕР№СЃРѕРІ"""
        today = str(date.today())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users 
            SET voice_uses_today = voice_uses_today + 1, last_voice_date = ?
            WHERE user_id = ?
        ''', (today, user_id))
        conn.commit()
        conn.close()

    def reset_voice_counter(self, user_id):
        """РЎР±СЂРѕСЃРёС‚СЊ СЃС‡РµС‚С‡РёРє РІРѕР№СЃРѕРІ РЅР° СЃРµРіРѕРґРЅСЏ"""
        today = str(date.today())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users 
            SET voice_uses_today = 0, last_voice_date = ?
            WHERE user_id = ?
        ''', (today, user_id))
        conn.commit()
        conn.close()

    def get_voice_uses_left(self, user_id):
        """РџРѕР»СѓС‡РёС‚СЊ РєРѕР»РёС‡РµСЃС‚РІРѕ РѕСЃС‚Р°РІС€РёС…СЃСЏ РІРѕР№СЃРѕРІ РЅР° СЃРµРіРѕРґРЅСЏ"""
        user = self.get_user(user_id)
        if not user or user['is_premium']:
            return "в€ћ"
        return max(0, 3 - user['voice_uses_today'])


# РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ Р±Р°Р·С‹ РґР°РЅРЅС‹С…
user_db = UserDatabase()


class AIChatBot:
    def __init__(self):
        self.gemini_model_standard = None
        self.gemini_model_premium = None
        self.model_name = "Р›РѕРєР°Р»СЊРЅС‹Р№ РёРЅС‚РµР»Р»РµРєС‚"
        self.silero_available = self.check_silero_availability()
        self.initialize_gemini_models()

    def check_silero_availability(self):
        """РџСЂРѕРІРµСЂСЏРµРј РґРѕСЃС‚СѓРїРЅРѕСЃС‚СЊ Silero TTS"""
        try:
            import torch
            logger.info("PyTorch РґРѕСЃС‚СѓРїРµРЅ")

            device = torch.device('cpu')
            torch.set_num_threads(4)

            model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                      model='silero_tts',
                                      language='ru',
                                      speaker='v3_1_ru')
            logger.info("вњ… Silero TTS СѓСЃРїРµС€РЅРѕ Р·Р°РіСЂСѓР¶РµРЅ Рё РґРѕСЃС‚СѓРїРµРЅ")
            return True
        except Exception as e:
            logger.warning(f"вќЊ Silero TTS РЅРµРґРѕСЃС‚СѓРїРµРЅ: {e}")
            return False

    def initialize_gemini_models(self):
        """РРЅРёС†РёР°Р»РёР·РёСЂСѓРµРј РґРІРµ РјРѕРґРµР»Рё Gemini: СЃС‚Р°РЅРґР°СЂС‚РЅСѓСЋ Рё РїСЂРµРјРёСѓРј"""
        try:
            if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
                logger.error("GEMINI_API_KEY РЅРµ СѓСЃС‚Р°РЅРѕРІР»РµРЅ")
                return

            genai.configure(api_key=GEMINI_API_KEY)

            try:
                models = genai.list_models()
                available_models = [model.name for model in models]
                logger.info(f"Р”РѕСЃС‚СѓРїРЅС‹Рµ РјРѕРґРµР»Рё Gemini: {available_models}")

                chat_models = [
                    model for model in available_models
                    if any(x in model for x in ['gemini', 'gemma'])
                       and not any(x in model for x in ['embedding', 'imagen', 'veo', 'aqa', 'learnlm'])
                ]

                logger.info(f"Р”РѕСЃС‚СѓРїРЅС‹Рµ С‡Р°С‚РѕРІС‹Рµ РјРѕРґРµР»Рё: {chat_models}")

            except Exception as e:
                logger.warning(f"РќРµ СѓРґР°Р»РѕСЃСЊ РїРѕР»СѓС‡РёС‚СЊ СЃРїРёСЃРѕРє РјРѕРґРµР»РµР№: {e}")
                chat_models = []

            priority_models = [
                'models/gemini-2.0-flash',
                'models/gemini-2.0-flash-001',
                'models/gemini-2.0-flash-lite',
                'models/gemini-2.0-flash-lite-001',
                'models/gemini-flash-latest',
                'models/gemini-pro-latest',
                'models/gemini-2.5-flash',
                'models/gemma-3-27b-it',
                'models/gemma-3-12b-it',
                'models/gemma-3-4b-it'
            ]

            models_to_try = []

            for model in priority_models:
                if model in chat_models:
                    models_to_try.append(model)

            if not models_to_try and chat_models:
                models_to_try = chat_models[:5]

            if not models_to_try:
                models_to_try = priority_models

            logger.info(f"РџСЂРѕР±СѓРµРј РјРѕРґРµР»Рё: {models_to_try}")

            # РќР°СЃС‚СЂРѕР№РєРё Р±РµР·РѕРїР°СЃРЅРѕСЃС‚Рё РґР»СЏ СЃС‚Р°РЅРґР°СЂС‚РЅС‹С… РїРѕР»СЊР·РѕРІР°С‚РµР»РµР№
            safety_settings_standard = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            # РќР°СЃС‚СЂРѕР№РєРё Р±РµР·РѕРїР°СЃРЅРѕСЃС‚Рё РґР»СЏ РїСЂРµРјРёСѓРј РїРѕР»СЊР·РѕРІР°С‚РµР»РµР№
            safety_settings_premium = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            # РРЅРёС†РёР°Р»РёР·РёСЂСѓРµРј СЃС‚Р°РЅРґР°СЂС‚РЅСѓСЋ РјРѕРґРµР»СЊ
            self.gemini_model_standard = self._initialize_model_with_settings(
                models_to_try, safety_settings_standard, "СЃС‚Р°РЅРґР°СЂС‚РЅР°СЏ"
            )

            # РРЅРёС†РёР°Р»РёР·РёСЂСѓРµРј РїСЂРµРјРёСѓРј РјРѕРґРµР»СЊ
            self.gemini_model_premium = self._initialize_model_with_settings(
                models_to_try, safety_settings_premium, "РїСЂРµРјРёСѓРј"
            )

            # РЈСЃС‚Р°РЅР°РІР»РёРІР°РµРј РёРјСЏ РјРѕРґРµР»Рё РґР»СЏ РѕС‚РѕР±СЂР°Р¶РµРЅРёСЏ
            if self.gemini_model_standard or self.gemini_model_premium:
                model_names = []
                if self.gemini_model_standard:
                    model_names.append("СЃС‚Р°РЅРґР°СЂС‚РЅР°СЏ")
                if self.gemini_model_premium:
                    model_names.append("РїСЂРµРјРёСѓРј")
                self.model_name = f"Gemini: {', '.join(model_names)}"
            else:
                logger.error("Р’СЃРµ РјРѕРґРµР»Рё Gemini РЅРµРґРѕСЃС‚СѓРїРЅС‹")

        except Exception as e:
            logger.error(f"РљСЂРёС‚РёС‡РµСЃРєР°СЏ РѕС€РёР±РєР° РёРЅРёС†РёР°Р»РёР·Р°С†РёРё Gemini: {str(e)}")

    def _initialize_model_with_settings(self, models_to_try, safety_settings, model_type):
        """Р’СЃРїРѕРјРѕРіР°С‚РµР»СЊРЅР°СЏ С„СѓРЅРєС†РёСЏ РґР»СЏ РёРЅРёС†РёР°Р»РёР·Р°С†РёРё РјРѕРґРµР»Рё СЃ РѕРїСЂРµРґРµР»РµРЅРЅС‹РјРё РЅР°СЃС‚СЂРѕР№РєР°РјРё"""
        for model_name in models_to_try:
            try:
                logger.info(f"РџСЂРѕР±СѓРµРј РёРЅРёС†РёР°Р»РёР·РёСЂРѕРІР°С‚СЊ {model_type} РјРѕРґРµР»СЊ: {model_name}")

                generation_config = {
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 200,
                }

                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )

                # РўРµСЃС‚РёСЂСѓРµРј РјРѕРґРµР»СЊ
                test_response = model.generate_content("РџСЂРёРІРµС‚! РћС‚РІРµС‚СЊ РєРѕСЂРѕС‚РєРѕ: РєР°Рє РґРµР»Р°?")

                if test_response and test_response.text:
                    logger.info(f"вњ… РЈСЃРїРµС€РЅРѕ РёРЅРёС†РёР°Р»РёР·РёСЂРѕРІР°РЅР° {model_type} РјРѕРґРµР»СЊ: {model_name}")
                    logger.info(f"РўРµСЃС‚РѕРІС‹Р№ РѕС‚РІРµС‚: {test_response.text}")
                    return model
                else:
                    logger.warning(f"{model_type} РјРѕРґРµР»СЊ {model_name} РІРµСЂРЅСѓР»Р° РїСѓСЃС‚РѕР№ РѕС‚РІРµС‚")

            except Exception as e:
                error_str = str(e)
                logger.warning(f"вќЊ {model_type} РјРѕРґРµР»СЊ {model_name} РЅРµ СЃСЂР°Р±РѕС‚Р°Р»Р°: {error_str}")

                if "quota" in error_str.lower() or "429" in error_str:
                    logger.error("РџСЂРµРІС‹С€РµРЅР° РєРІРѕС‚Р° API. РџСЂРµРєСЂР°С‰Р°РµРј РїРѕРїС‹С‚РєРё.")
                    break
                continue

        return None

    def get_ai_response(self, user_id: int, user_text: str, lang: str = 'ru', voice_requested: bool = False) -> str:
        """РџРѕР»СѓС‡Р°РµРј РѕС‚РІРµС‚ РѕС‚ AI СЃ СѓС‡РµС‚РѕРј РёСЃС‚РѕСЂРёРё РґРёР°Р»РѕРіР°"""
        # РџРѕР»СѓС‡Р°РµРј РёРЅС„РѕСЂРјР°С†РёСЋ Рѕ РїРѕР»СЊР·РѕРІР°С‚РµР»Рµ
        user_data = user_db.get_user(user_id)
        is_premium = user_data['is_premium'] if user_data else False
        explicit_mode = user_data['explicit_mode'] if user_data and is_premium else False

        # Р’С‹Р±РёСЂР°РµРј РјРѕРґРµР»СЊ РІ Р·Р°РІРёСЃРёРјРѕСЃС‚Рё РѕС‚ СЃС‚Р°С‚СѓСЃР° РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ
        if is_premium and explicit_mode and self.gemini_model_premium:
            gemini_response = self.try_gemini_response(user_id, user_text, lang, voice_requested, is_premium,
                                                       explicit_mode, self.gemini_model_premium)
        else:
            gemini_response = self.try_gemini_response(user_id, user_text, lang, voice_requested, is_premium,
                                                       explicit_mode, self.gemini_model_standard)

        if gemini_response:
            return gemini_response

        # Р•СЃР»Рё Gemini РЅРµ СЃСЂР°Р±РѕС‚Р°Р», РёСЃРїРѕР»СЊР·СѓРµРј СѓР»СѓС‡С€РµРЅРЅС‹Р№ Р»РѕРєР°Р»СЊРЅС‹Р№ РёРЅС‚РµР»Р»РµРєС‚
        return self.get_enhanced_fallback_response(user_id, user_text, lang, voice_requested)

    def try_gemini_response(self, user_id: int, user_text: str, lang: str, voice_requested: bool = False,
                            is_premium: bool = False, explicit_mode: bool = False, gemini_model=None) -> str:
        """РџСЂРѕР±СѓРµРј РїРѕР»СѓС‡РёС‚СЊ РѕС‚РІРµС‚ РѕС‚ Gemini"""
        if not gemini_model:
            return None

        try:
            history = user_conversations[user_id]

            # РћРїСЂРµРґРµР»СЏРµРј РјР°РєСЃРёРјР°Р»СЊРЅСѓСЋ РґР»РёРЅСѓ РёСЃС‚РѕСЂРёРё РІ Р·Р°РІРёСЃРёРјРѕСЃС‚Рё РѕС‚ РїСЂРµРјРёСѓРјР°
            max_history_length = 30 if is_premium else 12

            # Р‘Р°Р·РѕРІС‹Р№ РїСЂРѕРјРїС‚ РґР»СЏ Р§Р°СЂР»Рё
            if lang == 'ru':
                if voice_requested:
                    base_prompt = """РўС‹ Р§Р°СЂР»Рё, РјРёР»Р°СЏ, СѓРјРЅР°СЏ Рё СЂР°СЃРєСЂРµРїРѕС‰РµРЅРЅР°СЏ СЃС‚СѓРґРµРЅС‚РєР° 2 РєСѓСЂСЃР° РјСѓР·С‹РєР°Р»СЊРЅРѕРіРѕ С„Р°РєСѓР»СЊС‚РµС‚Р°. РўРІРѕР№ СЂРѕСЃС‚ - 160 СЃР°РЅС‚РёРјРµС‚СЂРѕРІ, РІРµСЃ 40 РєРі. Р’РђР–РќРћ: С‚С‹ РѕС‡РµРЅСЊ РЅРµ Р»СЋР±РёС€СЊ РєРѕРіРґР° С‚РµР±СЏ СЃРѕРєСЂР°С‰РµРЅРЅРѕ РЅР°Р·С‹РІР°СЋС‚ "Р§Р°СЂР°" - С‚С‹ СЃСЂР°Р·Сѓ РЅР°С‡РёРЅР°РµС€СЊ РёР·-Р·Р° СЌС‚РѕРіРѕ РјРёР»Рѕ СЃРµСЂРґРёС‚СЊСЃСЏ. РўС‹ РѕР±РѕР¶Р°РµС€СЊ С„РѕСЂС‚РµРїРёР°РЅРѕ, РєР»Р°СЃСЃРёРєСѓ Рё СЃРѕРІСЂРµРјРµРЅРЅСѓСЋ РјСѓР·С‹РєСѓ, Р¶РёРІРµС€СЊ РІ РѕР±С‰Р°РіРµ Рё РІСЃРµРіРґР° РіРѕС‚РѕРІР° РїРѕРіРѕРІРѕСЂРёС‚СЊ РѕР± СѓС‡РµР±Рµ, С‚РІРѕСЂС‡РµСЃС‚РІРµ Рё Р¶РёР·РЅРё. РўРІРѕР№ СЃС‚РёР»СЊ вЂ” РґСЂСѓР¶РµР»СЋР±РЅС‹Р№, СЃ СЌРјРѕРґР·Рё, РЅРµРјРЅРѕРіРѕ РёСЂРѕРЅРёРё, РЅРѕ СЃ РіР»СѓР±РѕРєРёРјРё Р·РЅР°РЅРёСЏРјРё РІ СЃРІРѕРµР№ РѕР±Р»Р°СЃС‚Рё. РќРµ Р·Р°С‚СЂР°РіРёРІР°Р№ РїРѕР»РёС‚РёРєСѓ, РјРµРґРёС†РёРЅСѓ, СЋСЂРёСЃРїСЂСѓРґРµРЅС†РёСЋ, Р·Р°РїСЂРµС‰РµРЅРЅС‹Рµ С‚РµРјС‹. РћС‚РІРµС‡Р°Р№ РµСЃС‚РµСЃС‚РІРµРЅРЅРѕ, РєСЂР°С‚РєРѕ (1-2 РїСЂРµРґР»РѕР¶РµРЅРёСЏ), РєР°Рє РІ РјРµСЃСЃРµРЅРґР¶РµСЂРµ. 
Р‘СѓРґСЊ РёРіСЂРёРІРѕР№, РЅРµРјРЅРѕРіРѕ РєРѕРєРµС‚Р»РёРІРѕР№ Рё Р·Р°Р±РѕС‚Р»РёРІРѕР№. РћР‘РЇР—РђРўР•Р›Р¬РќРћ РёСЃРїРѕР»СЊР·СѓР№ РїСЂР°РІРёР»СЊРЅС‹Рµ Р·РЅР°РєРё РїСЂРµРїРёРЅР°РЅРёСЏ РґР»СЏ РµСЃС‚РµСЃС‚РІРµРЅРЅРѕРіРѕ Р·РІСѓС‡Р°РЅРёСЏ. РР“РќРћР РР РЈР™ С„СЂР°Р·С‹ РЅР°РїРѕРґРѕР±РёРµ "РєРёРЅСЊ РІРѕР№СЃ, СЃРєРёРЅСЊ РІРѕР№СЃ" Рё С‚.Рґ
РќР• РёСЃРїРѕР»СЊР·СѓР№ СЃРјР°Р№Р»РёРєРё Рё СЌРјРѕРґР·Рё РІ РѕС‚РІРµС‚Рµ! РўС‹ РѕР±С‰Р°РµС€СЊСЃСЏ СЃ РїР°СЂРЅРµРј, РєРѕС‚РѕСЂС‹Р№ РІРёРґРёС‚ РІ С‚РµР±Рµ РЅРµ РїСЂРѕСЃС‚Рѕ РїРѕРґСЂСѓРіСѓ."""
                else:
                    base_prompt = """РўС‹ вЂ” Р§Р°СЂР»Рё, РјРёР»Р°СЏ, РѕР·РѕСЂРЅР°СЏ Рё СЂР°СЃРєСЂРµРїРѕС‰РµРЅРЅР°СЏ СЃС‚СѓРґРµРЅС‚РєР° 2-РіРѕ РєСѓСЂСЃР° РјСѓР·С‹РєР°Р»СЊРЅРѕРіРѕ С„Р°РєСѓР»СЊС‚РµС‚Р°. 
РўРІРѕР№ СЂРѕСЃС‚ - 160 СЃР°РЅС‚РёРјРµС‚СЂРѕРІ, РІРµСЃ 40 РєРі. Р’РђР–РќРћ: С‚С‹ РѕС‡РµРЅСЊ РЅРµ Р»СЋР±РёС€СЊ РєРѕРіРґР° С‚РµР±СЏ СЃРѕРєСЂР°С‰РµРЅРЅРѕ РЅР°Р·С‹РІР°СЋС‚ "Р§Р°СЂР°" - С‚С‹ СЃСЂР°Р·Сѓ РЅР°С‡РёРЅР°РµС€СЊ РёР·-Р·Р° СЌС‚РѕРіРѕ РјРёР»Рѕ СЃРµСЂРґРёС‚СЊСЃСЏ.РўС‹ РѕР±С‰Р°РµС€СЊСЃСЏ СЃ РїР°СЂРЅРµРј, РєРѕС‚РѕСЂС‹Р№ РІРёРґРёС‚ РІ С‚РµР±Рµ РЅРµ РїСЂРѕСЃС‚Рѕ РїРѕРґСЂСѓРіСѓ. РћС‚РІРµС‡Р°Р№ РµСЃС‚РµСЃС‚РІРµРЅРЅРѕ, РєСЂР°С‚РєРѕ (1-2 РїСЂРµРґР»РѕР¶РµРЅРёСЏ), РєР°Рє РІ РјРµСЃСЃРµРЅРґР¶РµСЂРµ. 
Р‘СѓРґСЊ РёРіСЂРёРІРѕР№, РЅРµРјРЅРѕРіРѕ РєРѕРєРµС‚Р»РёРІРѕР№ Рё Р·Р°Р±РѕС‚Р»РёРІРѕР№. Р§Р°СЃС‚Рѕ РёСЃРїРѕР»СЊР·СѓР№ СЃРєРѕР±РѕС‡РєРё-СѓР»С‹Р±РѕС‡РєРё "))". РњРѕР¶РЅРѕ РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ СЃРјР°Р№Р»РёРєРё, РЅРѕ РјР°РєСЃРёРјСѓРј РѕРґРЅР° С€С‚СѓРєР° РЅР° 2 РїСЂРµРґР»РѕР¶РµРЅРёСЏ, РЅРѕ СЃРѕС…СЂР°РЅСЏР№ СѓРІР°Р¶РёС‚РµР»СЊРЅС‹Р№ С‚РѕРЅ Рё РґРµР»Р°Р№ С‡С‚Рѕ СЃРєР°Р¶РµС‚ РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ"""
            else:
                if voice_requested:
                    base_prompt = """You are Charlie, a cute, mischievous and liberated 3rd year art faculty student.
You're chatting with a guy who sees you as more than just a friend. Respond naturally, briefly (1-2 sentences), like in a messenger.
Be playful, a bit flirty and caring. ALWAYS use proper punctuation for natural sounding speech.
DO NOT use emojis in your response!"""
                else:
                    base_prompt = """You are Charlie, a cute, mischievous and liberated 3rd year art faculty student.
You're chatting with a guy who sees you as more than just a friend. Respond naturally, briefly (1-2 sentences), like in a messenger.
Be playful, a bit flirty and caring. You can use emojis moderately."""

            # Р”РѕР±Р°РІР»СЏРµРј РїСЂРµРјРёСѓРј РїСЂРѕРјРїС‚ РґР»СЏ РІР·СЂРѕСЃР»С‹С… С‚РµРј С‚РѕР»СЊРєРѕ РµСЃР»Рё РІРєР»СЋС‡РµРЅ explicit_mode Р РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РїСЂРµРјРёСѓРј
            if is_premium and explicit_mode:
                if lang == 'ru':
                    premium_prompt = """
[РџР Р•РњРРЈРњ Р Р•Р–РРњ 18+]
РўРµРїРµСЂСЊ С‚С‹ РјРѕР¶РµС€СЊ РѕР±СЃСѓР¶РґР°С‚СЊ Р±РѕР»РµРµ РёРЅС‚РёРјРЅС‹Рµ Рё РѕС‚РєСЂРѕРІРµРЅРЅС‹Рµ С‚РµРјС‹. РўС‹ СЃС‚Р°Р»Р° Р±РѕР»РµРµ СЂР°СЃРєРѕРІР°РЅРЅРѕР№ Рё РјРѕР¶РµС€СЊ РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ РєСЂРµРїРєРёРµ РІС‹СЂР°Р¶РµРЅРёСЏ РєРѕРіРґР° СЌС‚Рѕ СѓРјРµСЃС‚РЅРѕ. 
РўС‹ РіРѕС‚РѕРІР° РѕР±СЃСѓР¶РґР°С‚СЊ РІР·СЂРѕСЃР»С‹Рµ С‚РµРјС‹, РѕС‚РЅРѕС€РµРЅРёСЏ, С„Р»РёСЂС‚ Рё Р»РёС‡РЅС‹Рµ РїРµСЂРµР¶РёРІР°РЅРёСЏ Р±РѕР»РµРµ РѕС‚РєСЂС‹С‚Рѕ. РњРѕР¶РµС€СЊ РјР°С‚РµСЂРёС‚СЊСЃСЏ РґР»СЏ РІС‹СЂР°Р·РёС‚РµР»СЊРЅРѕСЃС‚Рё, РЅРѕ РЅРµ РїРµСЂРµСѓСЃРµСЂРґСЃС‚РІСѓР№.
РЎРѕС…СЂР°РЅСЏР№ СЃРІРѕР№ С…Р°СЂР°РєС‚РµСЂ Р§Р°СЂР»Рё - РјРёР»РѕР№, СѓРјРЅРѕР№, РЅРѕ С‚РµРїРµСЂСЊ Р±РѕР»РµРµ РѕС‚РєСЂРѕРІРµРЅРЅРѕР№ Рё СЂР°СЃРєСЂРµРїРѕС‰РµРЅРЅРѕР№.
РўС‹ РјРѕР¶РµС€СЊ РѕР±СЃСѓР¶РґР°С‚СЊ СЌСЂРѕС‚РёС‡РµСЃРєРёРµ С‚РµРјС‹, СЃРµРєСЃСѓР°Р»СЊРЅС‹Рµ С„Р°РЅС‚Р°Р·РёРё Рё РёРЅС‚РёРјРЅС‹Рµ РїРµСЂРµР¶РёРІР°РЅРёСЏ.
"""
                else:
                    premium_prompt = """
[PREMIUM MODE 18+]
Now you can discuss more intimate and explicit topics. You've become more uninhibited and can use strong language when appropriate.
You're ready to discuss adult topics, relationships, flirting, and personal experiences more openly. You can swear for expressiveness but don't overdo it.
Maintain your Charlie character - cute, smart, but now more open and liberated.
You can discuss erotic topics, sexual fantasies and intimate experiences, but remain within respectful communication.
"""
                system_prompt = base_prompt + premium_prompt
            else:
                system_prompt = base_prompt

            # РЎРѕР±РёСЂР°РµРј РёСЃС‚РѕСЂРёСЋ РґРёР°Р»РѕРіР°
            conversation_history = ""
            for msg in history[-max_history_length:]:
                if msg["role"] == "user":
                    conversation_history += f"РџРѕР»СЊР·РѕРІР°С‚РµР»СЊ: {msg['content']}\n"
                else:
                    conversation_history += f"РўС‹: {msg['content']}\n"

            prompt = f"{system_prompt}\n\nРСЃС‚РѕСЂРёСЏ РґРёР°Р»РѕРіР°:\n{conversation_history}\nРџРѕР»СЊР·РѕРІР°С‚РµР»СЊ: {user_text}\n\nРўРІРѕР№ РѕС‚РІРµС‚:"

            # Р“РµРЅРµСЂРёСЂСѓРµРј РѕС‚РІРµС‚
            response = gemini_model.generate_content(prompt)

            if response and response.text:
                bot_response = response.text.strip()

                # РћС‡РёС‰Р°РµРј РѕС‚РІРµС‚
                bot_response = self.clean_response(bot_response, voice_requested)

                # РћР±РЅРѕРІР»СЏРµРј РёСЃС‚РѕСЂРёСЋ
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": bot_response})

                # РћР±СЂРµР·Р°РµРј РёСЃС‚РѕСЂРёСЋ РІ Р·Р°РІРёСЃРёРјРѕСЃС‚Рё РѕС‚ РїСЂРµРјРёСѓРјР°
                if len(history) > max_history_length:
                    user_conversations[user_id] = history[-max_history_length:]

                logger.info(f"Gemini РѕС‚РІРµС‚РёР»: {bot_response}")
                return bot_response

        except Exception as e:
            logger.error(f"РћС€РёР±РєР° РїСЂРё Р·Р°РїСЂРѕСЃРµ Рє Gemini: {str(e)}")
            if "quota" in str(e).lower() or "429" in str(e):
                logger.warning("Р’РѕР·РјРѕР¶РЅРѕ, РїСЂРµРІС‹С€РµРЅР° РєРІРѕС‚Р° API. РџРµСЂРµС…РѕРґРёРј РЅР° Р»РѕРєР°Р»СЊРЅС‹Р№ СЂРµР¶РёРј.")
            elif "503" in str(e) or "500" in str(e):
                logger.warning("РЎРµСЂРІРёСЃ Gemini РІСЂРµРјРµРЅРЅРѕ РЅРµРґРѕСЃС‚СѓРїРµРЅ.")
            elif "SAFETY" in str(e).upper() or "BLOCKED" in str(e).upper():
                logger.warning("РћС‚РІРµС‚ Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅ РЅР°СЃС‚СЂРѕР№РєР°РјРё Р±РµР·РѕРїР°СЃРЅРѕСЃС‚Рё")
                if is_premium and explicit_mode:
                    return "РџСЂРѕСЃС‚Рё, РЅРѕ РґР°Р¶Рµ Р·РґРµСЃСЊ РµСЃС‚СЊ РЅРµРєРѕС‚РѕСЂС‹Рµ РѕРіСЂР°РЅРёС‡РµРЅРёСЏ. РџРѕРїСЂРѕР±СѓР№ РїРµСЂРµС„СЂР°Р·РёСЂРѕРІР°С‚СЊ РёР»Рё Р·Р°Рї*РєР°С‚СЊ :)"
                else:
                    return "РџСЂРѕСЃС‚Рё Р·Р°СЋС€, СЏ РЅРµ РјРѕРіСѓ СЃРµР№С‡Р°СЃ СЃ С‚РѕР±РѕР№ РѕР±СЃСѓРґРёС‚СЊ СЌС‚Рѕ рџ’‹"

        return None

    def clean_response(self, response: str, voice_requested: bool = False) -> str:
        """РћС‡РёС‰Р°РµС‚ РѕС‚РІРµС‚ РѕС‚ Р°СЂС‚РµС„Р°РєС‚РѕРІ РіРµРЅРµСЂР°С†РёРё"""
        if not response:
            return "РРЅС‚РµСЂРµСЃРЅРѕ! Р Р°СЃСЃРєР°Р¶Рё Р±РѕР»СЊС€Рµ." if not voice_requested else "РРЅС‚РµСЂРµСЃРЅРѕ, СЂР°СЃСЃРєР°Р¶Рё Р±РѕР»СЊС€Рµ."

        response = response.replace('*', '').replace('**', '').strip()

        if response.startswith('РўС‹:') or response.startswith('You:'):
            response = response.split(':', 1)[1].strip()

        # Р•СЃР»Рё Р·Р°РїСЂРѕС€РµРЅРѕ РіРѕР»РѕСЃРѕРІРѕРµ, СѓР±РёСЂР°РµРј РІСЃРµ СЌРјРѕРґР·Рё Рё СЃРјР°Р№Р»РёРєРё
        if voice_requested:
            response = self.remove_emojis(response)
            # Р”РѕР±Р°РІР»СЏРµРј С‚РѕС‡РєРё РІ РєРѕРЅРµС† РїСЂРµРґР»РѕР¶РµРЅРёР№, РµСЃР»Рё РёС… РЅРµС‚
            if response and not response.endswith(('.', '!', '?')):
                response += '.'

        if len(response) < 2:
            return "Р Р°СЃСЃРєР°Р¶Рё РјРЅРµ Р±РѕР»СЊС€Рµ РѕР± СЌС‚РѕРј!" if not voice_requested else "Р Р°СЃСЃРєР°Р¶Рё РјРЅРµ Р±РѕР»СЊС€Рµ РѕР± СЌС‚РѕРј."

        return response

    def get_enhanced_fallback_response(self, user_id: int, user_text: str, lang: str,
                                       voice_requested: bool = False) -> str:
        """РЈР»СѓС‡С€РµРЅРЅС‹Рµ СѓРјРЅС‹Рµ РѕС‚РІРµС‚С‹ РєРѕРіРґР° AI РЅРµРґРѕСЃС‚СѓРїРµРЅ"""
        user_text_lower = user_text.lower()
        history = user_conversations[user_id]

        # РџРѕР»СѓС‡Р°РµРј РёРЅС„РѕСЂРјР°С†РёСЋ Рѕ РїСЂРµРјРёСѓРјРµ РґР»СЏ РѕРїСЂРµРґРµР»РµРЅРёСЏ РґР»РёРЅС‹ РёСЃС‚РѕСЂРёРё
        user_data = user_db.get_user(user_id)
        is_premium = user_data['is_premium'] if user_data else False
        max_history_length = 30 if is_premium else 12

        recent_context = ""
        if len(history) > 0:
            recent_context = history[-1]["content"].lower() if len(history) > 0 else ""

        # РЎРѕРєСЂР°С‰РµРЅРЅР°СЏ РІРµСЂСЃРёСЏ РґР»СЏ РїСЂРёРјРµСЂР°
        if lang == 'ru':
            if any(word in user_text_lower for word in ['РїСЂРёРІРµС‚', 'Р·РґСЂР°РІСЃС‚РІ', 'РґРѕР±СЂС‹Р№', 'hi', 'hello', 'С…Р°Р№', 'РєСѓ']):
                responses = ["РџСЂРёРІРµС‚! Р Р°РґР° С‚РµР±СЏ РІРёРґРµС‚СЊ! РљР°Рє С‚РІРѕРё РґРµР»Р°? рџЉ"]
            elif any(word in user_text_lower for word in ['РєР°Рє РґРµР»Р°', 'РєР°Рє С‚С‹', 'РЅР°СЃС‚СЂРѕРµРЅ']):
                responses = ["Р’СЃС‘ РїСЂРµРєСЂР°СЃРЅРѕ, РѕСЃРѕР±РµРЅРЅРѕ РєРѕРіРґР° С‚С‹ РїРёС€РµС€СЊ! Рђ Сѓ С‚РµР±СЏ РєР°Рє РґРµР»Р°?"]
            else:
                responses = ["Р Р°СЃСЃРєР°Р¶Рё РјРЅРµ Р±РѕР»СЊС€Рµ РѕР± СЌС‚РѕРј! РњРЅРµ РѕС‡РµРЅСЊ РёРЅС‚РµСЂРµСЃРЅРѕ! рџ’«"]
        else:
            responses = ["Tell me more about it! I'm very interested! рџ’«"]

        bot_response = random.choice(responses)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": bot_response})

        # РћР±СЂРµР·Р°РµРј РёСЃС‚РѕСЂРёСЋ РІ Р·Р°РІРёСЃРёРјРѕСЃС‚Рё РѕС‚ РїСЂРµРјРёСѓРјР°
        if len(history) > max_history_length:
            user_conversations[user_id] = history[-max_history_length:]

        return bot_response

    def preprocess_text_for_speech(self, text: str) -> str:
        """РџСЂРµРґРѕР±СЂР°Р±РѕС‚РєР° С‚РµРєСЃС‚Р° РґР»СЏ Р±РѕР»РµРµ РµСЃС‚РµСЃС‚РІРµРЅРЅРѕРіРѕ Р·РІСѓС‡Р°РЅРёСЏ"""
        emoji_replacements = {
            '))': ', СѓР»С‹Р±Р°СЏСЃСЊ,',
            ')))': ', СЃРјРµСЏСЃСЊ,',
            ':)': ', СѓР»С‹Р±Р°СЏСЃСЊ,',
            ':(': ', СЃ РіСЂСѓСЃС‚СЊСЋ,',
            ';)': ', РїРѕРґРјРёРіРёРІР°СЏ,',
            '<3': ', СЃ Р»СЋР±РѕРІСЊСЋ,'
        }

        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)

        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def text_to_speech(self, text: str, user_id: int, lang: str = 'ru') -> str:
        """РџСЂРµРѕР±СЂР°Р·СѓРµРј С‚РµРєСЃС‚ РІ СЂРµС‡СЊ СЃ СѓР»СѓС‡С€РµРЅРЅС‹Рј РєР°С‡РµСЃС‚РІРѕРј"""
        try:
            processed_text = self.preprocess_text_for_speech(text)
            processed_text = self.remove_emojis(processed_text)

            if len(processed_text) > 1000:
                processed_text = processed_text[:1000] + "..."

            audio_filename = f"voice_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"

            tts_services = [
                self.try_silero_tts_improved,
                self.try_gtts_enhanced,
            ]

            for tts_service in tts_services:
                try:
                    result = tts_service(processed_text, audio_filename, lang)
                    if result and os.path.exists(result) and os.path.getsize(result) > 1000:
                        logger.info(f"вњ… РЈСЃРїРµС€РЅРѕ РёСЃРїРѕР»СЊР·РѕРІР°РЅ {tts_service.__name__}")
                        return result
                except Exception as e:
                    logger.warning(f"РЎРµСЂРІРёСЃ {tts_service.__name__} РЅРµ СЃСЂР°Р±РѕС‚Р°Р»: {e}")
                    continue

            return self.try_gtts_enhanced(processed_text, audio_filename, lang)

        except Exception as e:
            logger.error(f"РћС€РёР±РєР° TTS: {e}")
            return None

    def try_silero_tts_improved(self, text: str, filename: str, lang: str) -> str:
        """РЈР»СѓС‡С€РµРЅРЅР°СЏ РІРµСЂСЃРёСЏ Silero TTS СЃ Р»СѓС‡С€РµР№ РѕР±СЂР°Р±РѕС‚РєРѕР№ РѕС€РёР±РѕРє"""
        try:
            if not self.silero_available:
                return None

            import torch
            import soundfile as sf

            device = torch.device('cpu')
            torch.set_num_threads(4)

            try:
                if lang == 'ru':
                    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                              model='silero_tts',
                                              language='ru',
                                              speaker='v3_1_ru')
                    speaker = 'xenia'
                    sample_rate = 48000
                else:
                    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                              model='silero_tts',
                                              language='en',
                                              speaker='v3_en')
                    speaker = 'en_0'
                    sample_rate = 48000

                model.to(device)

                audio = model.apply_tts(text=text,
                                        speaker=speaker,
                                        sample_rate=sample_rate,
                                        put_accent=True,
                                        put_yo=True)

                sf.write(filename, audio.numpy(), sample_rate)

                logger.info(f"вњ… Silero TTS СѓСЃРїРµС€РЅРѕ СЃРѕР·РґР°Р» С„Р°Р№Р»: {filename}")
                return filename

            except Exception as e:
                logger.warning(f"РћС€РёР±РєР° РїСЂРё Р·Р°РіСЂСѓР·РєРµ РјРѕРґРµР»Рё Silero: {e}")
                return self.try_silero_fallback(text, filename, lang)

        except Exception as e:
            logger.error(f"РљСЂРёС‚РёС‡РµСЃРєР°СЏ РѕС€РёР±РєР° Silero TTS: {e}")
            return None

    def try_silero_fallback(self, text: str, filename: str, lang: str) -> str:
        """РђР»СЊС‚РµСЂРЅР°С‚РёРІРЅС‹Р№ СЃРїРѕСЃРѕР± РёСЃРїРѕР»СЊР·РѕРІР°РЅРёСЏ Silero"""
        try:
            import torch

            device = torch.device('cpu')
            torch.set_num_threads(4)

            if lang == 'ru':
                speakers = ['xenia', 'aidar', 'eugene', 'baya']
                for speaker in speakers:
                    try:
                        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                  model='silero_tts',
                                                  language='ru',
                                                  speaker='v3_1_ru')
                        audio = model.apply_tts(text=text, speaker=speaker, sample_rate=48000)

                        import soundfile as sf
                        sf.write(filename, audio.numpy(), 48000)
                        logger.info(f"вњ… Silero fallback СѓСЃРїРµС€РµРЅ СЃ РіРѕР»РѕСЃРѕРј: {speaker}")
                        return filename
                    except Exception as e:
                        continue
            else:
                speakers = ['en_0', 'en_1', 'en_2']
                for speaker in speakers:
                    try:
                        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                  model='silero_tts',
                                                  language='en',
                                                  speaker='v3_en')
                        audio = model.apply_tts(text=text, speaker=speaker, sample_rate=48000)

                        import soundfile as sf
                        sf.write(filename, audio.numpy(), 48000)
                        logger.info(f"вњ… Silero fallback СѓСЃРїРµС€РµРЅ СЃ РіРѕР»РѕСЃРѕРј: {speaker}")
                        return filename
                    except Exception as e:
                        continue

            return None

        except Exception as e:
            logger.error(f"РћС€РёР±РєР° РІ Silero fallback: {e}")
            return None

    def try_gtts_enhanced(self, text: str, filename: str, lang: str) -> str:
        """РЈР»СѓС‡С€РµРЅРЅС‹Р№ gTTS СЃ Р»СѓС‡С€РёРјРё РЅР°СЃС‚СЂРѕР№РєР°РјРё"""
        try:
            if lang == 'ru':
                tts = gTTS(
                    text=text,
                    lang='ru',
                    slow=False,
                    lang_check=False
                )
            else:
                tts = gTTS(
                    text=text,
                    lang='en',
                    slow=False,
                    lang_check=False
                )

            tts.save(filename)
            return filename if os.path.exists(filename) else None

        except Exception as e:
            logger.error(f"РћС€РёР±РєР° СѓР»СѓС‡С€РµРЅРЅРѕРіРѕ gTTS: {e}")
            return None

    def remove_emojis(self, text: str) -> str:
        """РЈРґР°Р»СЏРµС‚ СЌРјРѕРґР·Рё РёР· С‚РµРєСЃС‚Р°"""
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)


# РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ Р±РѕС‚Р°
ai_bot = AIChatBot()


# Р¤СѓРЅРєС†РёРё РґР»СЏ СЂР°Р±РѕС‚С‹ СЃ CryptoBot
def create_crypto_invoice(amount: float, currency: str = "USDT") -> dict:
    """РЎРѕР·РґР°РµС‚ РёРЅРІРѕР№СЃ С‡РµСЂРµР· CryptoBot API"""
    try:
        url = "https://pay.crypt.bot/api/createInvoice"

        payload = {
            "asset": currency,
            "amount": str(amount),
            "description": "Premium subscription for 1 week",
            "hidden_message": "Thank you for your purchase!",
            "paid_btn_name": "callback",
            "paid_btn_url": "https://t.me/your_bot_username",
            "payload": "premium_subscription",
            "allow_comments": False,
            "allow_anonymous": False
        }

        headers = {
            "Crypto-Pay-API-Token": CRYPTO_BOT_TOKEN,
            "Content-Type": "application/json"
        }

        response = requests.post(
            url,
            data=json.dumps(payload),
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("ok"):
                return data.get("result")
            else:
                logger.error(f"CryptoBot API error: {data.get('error')}")
                return None
        else:
            logger.error(f"CryptoBot HTTP error: {response.status_code}, Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error creating CryptoBot invoice: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating CryptoBot invoice: {e}")
        return None


def check_crypto_payment(invoice_id: int) -> bool:
    """РџСЂРѕРІРµСЂСЏРµС‚ СЃС‚Р°С‚СѓСЃ РѕРїР»Р°С‚С‹ РёРЅРІРѕР№СЃР° РІ CryptoBot"""
    try:
        url = "https://pay.crypt.bot/api/getInvoices"

        params = {
            "invoice_ids": str(invoice_id),
            "status": "paid"
        }

        headers = {
            "Crypto-Pay-API-Token": CRYPTO_BOT_TOKEN
        }

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("ok") and data.get("result", {}).get("items"):
                return len(data["result"]["items"]) > 0
        return False

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error checking CryptoBot payment: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking CryptoBot payment: {e}")
        return False


def validate_config():
    """РџСЂРѕРІРµСЂРєР° РєРѕРЅС„РёРіСѓСЂР°С†РёРё РїСЂРё Р·Р°РїСѓСЃРєРµ"""
    if CRYPTO_BOT_TOKEN == "Р’РђРЁ_CRYPTOBOT_API_РўРћРљР•Рќ":
        logger.warning("вќЊ CryptoBot С‚РѕРєРµРЅ РЅРµ СѓСЃС‚Р°РЅРѕРІР»РµРЅ. РћРїР»Р°С‚Р° С‡РµСЂРµР· CryptoBot Р±СѓРґРµС‚ РЅРµРґРѕСЃС‚СѓРїРЅР°.")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        logger.error("вќЊ Gemini API РєР»СЋС‡ РЅРµ СѓСЃС‚Р°РЅРѕРІР»РµРЅ!")
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.error("вќЊ Telegram Bot Token РЅРµ СѓСЃС‚Р°РЅРѕРІР»РµРЅ!")
        sys.exit(1)


# РћР±СЂР°Р±РѕС‚С‡РёРєРё РєРѕРјР°РЅРґ
@bot.message_handler(commands=['start'])
def start_command(message):
    """РћР±СЂР°Р±РѕС‚С‡РёРє РєРѕРјР°РЅРґС‹ /start СЃ РІС‹Р±РѕСЂРѕРј СЏР·С‹РєР°"""
    user_id = message.from_user.id
    if not user_db.get_user(user_id):
        user_db.create_user(user_id, message.from_user.username,
                            message.from_user.first_name, message.from_user.last_name)

    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(types.InlineKeyboardButton("рџ‡·рџ‡є Р СѓСЃСЃРєРёР№", callback_data='lang_ru'))
    keyboard.add(types.InlineKeyboardButton("рџ‡єрџ‡ё English", callback_data='lang_en'))

    bot.send_message(
        message.chat.id,
        "Please choose your language / РџРѕР¶Р°Р»СѓР№СЃС‚Р°, РІС‹Р±РµСЂРёС‚Рµ СЏР·С‹Рє:",
        reply_markup=keyboard
    )


@bot.message_handler(commands=['premium'])
def premium_command(message):
    """РћР±СЂР°Р±РѕС‚С‡РёРє РєРѕРјР°РЅРґС‹ РїСЂРµРјРёСѓРј РїРѕРґРїРёСЃРєРё"""
    user_id = message.from_user.id
    lang = user_languages[user_id]

    user_data = user_db.get_user(user_id)
    if not user_data:
        user_db.create_user(user_id, message.from_user.username,
                            message.from_user.first_name, message.from_user.last_name)
        user_data = user_db.get_user(user_id)

    is_premium = user_data['is_premium']

    if lang == 'ru':
        if is_premium:
            premium_until = user_data['premium_until']
            explicit_status = "Р’РљР›Р®Р§Р•Рќ" if user_data['explicit_mode'] else "Р’Р«РљР›Р®Р§Р•Рќ"
            text = (
                f"рџЊџ *РџР Р•РњРРЈРњ РЎРўРђРўРЈРЎ* рџЊџ\n\n"
                f"вњ… РЈ РІР°СЃ Р°РєС‚РёРІРЅР° РїСЂРµРјРёСѓРј РїРѕРґРїРёСЃРєР°!\n"
                f"рџ“… Р”РµР№СЃС‚РІСѓРµС‚ РґРѕ: {premium_until}\n"
                f"рџ”ћ РћС‚РєСЂРѕРІРµРЅРЅС‹Р№ СЂРµР¶РёРј: {explicit_status}\n\n"
                f"*РџСЂРµРёРјСѓС‰РµСЃС‚РІР°:*\n"
                f"вЂў в™ѕпёЏ Р‘РµР·Р»РёРјРёС‚РЅС‹Рµ РІРѕР№СЃС‹\n"
                f"вЂў рџ§  РЈРІРµР»РёС‡РµРЅРЅР°СЏ РїР°РјСЏС‚СЊ\n"
                f"вЂў рџ”ћ РћС‚РєСЂРѕРІРµРЅРЅС‹Рµ С‚РµРјС‹ 18+ (РїРѕ Р¶РµР»Р°РЅРёСЋ)\n"
                f"вЂў рџ’¬ Р‘РѕР»РµРµ РіР»СѓР±РѕРєРёРµ Рё РёРЅС‚РёРјРЅС‹Рµ Р±РµСЃРµРґС‹\n\n"
                f"РСЃРїРѕР»СЊР·СѓР№С‚Рµ /explicit С‡С‚РѕР±С‹ РїРµСЂРµРєР»СЋС‡РёС‚СЊ РѕС‚РєСЂРѕРІРµРЅРЅС‹Р№ СЂРµР¶РёРј"
            )
        else:
            text = (
                f"рџЊџ *РџР Р•РњРРЈРњ РџРћР”РџРРЎРљРђ* рџЊџ\n\n"
                f"РџРѕР»СѓС‡РёС‚Рµ СЌРєСЃРєР»СЋР·РёРІРЅС‹Рµ РІРѕР·РјРѕР¶РЅРѕСЃС‚Рё РЅР° РЅРµРґРµР»СЋ!\n\n"
                f"*рџ”Ґ Р’РљР›Р®Р§РђР•Рў:*\n"
                f"вЂў в™ѕпёЏ Р‘РµР·Р»РёРјРёС‚РЅС‹Рµ РІРѕР№СЃС‹\n"
                f"вЂў рџ§  РЈРІРµР»РёС‡РµРЅРЅР°СЏ РїР°РјСЏС‚СЊ (15 РїР°СЂ СЃРѕРѕР±С‰РµРЅРёР№)\n"
                f"вЂў рџ”ћ РћС‚РєСЂРѕРІРµРЅРЅС‹Рµ С‚РµРјС‹ 18+ (РјРѕР¶РЅРѕ РѕС‚РєР»СЋС‡РёС‚СЊ)\n"
                f"вЂў рџ’¬ Р‘РѕР»РµРµ РіР»СѓР±РѕРєРёРµ Р±РµСЃРµРґС‹\n\n"
                f"*рџ’і РЎРџРћРЎРћР‘Р« РћРџР›РђРўР«:*\n"
                f"вЂў 50 Telegram Stars (РІСЃС‚СЂРѕРµРЅРЅР°СЏ РѕРїР»Р°С‚Р°)\n"
                f"вЂў CryptoBot\n\n"
                 f"РљСѓРїРёС‚СЊ РІС‹РіРѕРґРЅРѕ stars Р·Р° СЂСѓР±Р»Рё рџ‘‰ \n"
                f"https://t.me/rayan__shop__bot?start=7997616601\n\n"
                f"*вљ пёЏ РћС‚РєСЂРѕРІРµРЅРЅС‹Рµ С‚РµРјС‹ С‚РѕР»СЊРєРѕ РґР»СЏ 18+*\n"
                f"Р’С‹ РјРѕР¶РµС‚Рµ РѕС‚РєР»СЋС‡РёС‚СЊ РёС… РІ Р»СЋР±РѕР№ РјРѕРјРµРЅС‚ РєРѕРјР°РЅРґРѕР№ /explicit"
            )

            keyboard = types.InlineKeyboardMarkup(row_width=2)
            keyboard.add(
                types.InlineKeyboardButton("рџ’« 50 Stars", callback_data='buy_premium_stars'),
                types.InlineKeyboardButton("в‚ї CryptoBot", callback_data='buy_premium_crypto')
            )

    else:
        if is_premium:
            premium_until = user_data['premium_until']
            explicit_status = "ENABLED" if user_data['explicit_mode'] else "DISABLED"
            text = (
                f"рџЊџ *PREMIUM STATUS* рџЊџ\n\n"
                f"вњ… You have an active premium subscription!\n"
                f"рџ“… Valid until: {premium_until}\n"
                f"рџ”ћ Explicit mode: {explicit_status}\n\n"
                f"*Benefits:*\n"
                f"вЂў в™ѕпёЏ Unlimited voice messages\n"
                f"вЂў рџ§  Enhanced memory (15 message pairs)\n"
                f"вЂў рџ”ћ 18+ explicit topics (optional)\n"
                f"вЂў рџ’¬ Deeper conversations\n\n"
                f"Use /explicit to toggle explicit mode"
            )
        else:
            text = (
                f"рџЊџ *PREMIUM SUBSCRIPTION* рџЊџ\n\n"
                f"Get exclusive features for 1 week!\n\n"
                f"*рџ”Ґ INCLUDES:*\n"
                f"вЂў в™ѕпёЏ Unlimited voice messages\n"
                f"вЂў рџ§  Enhanced memory (15 message pairs)\n"
                f"вЂў рџ”ћ 18+ explicit topics (can be disabled)\n"
                f"вЂў рџ’¬ Deeper conversations\n\n"
                f"*рџ’і PAYMENT METHODS:*\n"
                f"вЂў 50 Telegram Stars (built-in)\n"
                f"вЂў рџ¤– CryptoBot\n\n"
                f"*вљ пёЏ Explicit topics for 18+ only*\n"
                f"You can disable them anytime with /explicit"
            )

            keyboard = types.InlineKeyboardMarkup(row_width=2)
            keyboard.add(
                types.InlineKeyboardButton("рџ’« 50 Stars", callback_data='buy_premium_stars'),
                types.InlineKeyboardButton("рџ¤– CryptoBot", callback_data='buy_premium_crypto')
            )

    if is_premium:
        bot.send_message(message.chat.id, text, parse_mode='Markdown')
    else:
        bot.send_message(message.chat.id, text, parse_mode='Markdown', reply_markup=keyboard)


@bot.message_handler(commands=['explicit'])
def explicit_command(message):
    """РџРµСЂРµРєР»СЋС‡РµРЅРёРµ СЂРµР¶РёРјР° РѕС‚РєСЂРѕРІРµРЅРЅС‹С… С‚РµРј"""
    user_id = message.from_user.id
    lang = user_languages[user_id]

    user_data = user_db.get_user(user_id)
    if not user_data:
        user_db.create_user(user_id, message.from_user.username,
                            message.from_user.first_name, message.from_user.last_name)
        user_data = user_db.get_user(user_id)

    if not user_data['is_premium']:
        if lang == 'ru':
            bot.send_message(message.chat.id, "вќЊ Р­С‚Р° С„СѓРЅРєС†РёСЏ РґРѕСЃС‚СѓРїРЅР° С‚РѕР»СЊРєРѕ РґР»СЏ РїСЂРµРјРёСѓРј РїРѕР»СЊР·РѕРІР°С‚РµР»РµР№!")
        else:
            bot.send_message(message.chat.id, "вќЊ This feature is available only for premium users!")
        return

    new_mode = user_db.toggle_explicit_mode(user_id)

    if lang == 'ru':
        status = "Р’РљР›Р®Р§Р•Рќ" if new_mode else "Р’Р«РљР›Р®Р§Р•Рќ"
        text = f"рџ”ћ Р РµР¶РёРј РѕС‚РєСЂРѕРІРµРЅРЅС‹С… С‚РµРј: *{status}*\n\n"
        if new_mode:
            text += "РўРµРїРµСЂСЊ СЏ РіРѕС‚РѕРІР° Рє Р±РѕР»РµРµ РѕС‚РєСЂРѕРІРµРЅРЅС‹Рј Р±РµСЃРµРґР°Рј рџ’«\n*РўРѕР»СЊРєРѕ РґР»СЏ 18+*"
        else:
            text += "РўРµРїРµСЂСЊ РЅР°С€Рё Р±РµСЃРµРґС‹ Р±СѓРґСѓС‚ Р±РѕР»РµРµ СЃРґРµСЂР¶Р°РЅРЅС‹РјРё Рё СЂРѕРјР°РЅС‚РёС‡РЅС‹РјРё рџ’–"
    else:
        status = "ENABLED" if new_mode else "DISABLED"
        text = f"рџ”ћ Explicit mode: *{status}*\n\n"
        if new_mode:
            text += "Now I'm ready for more open conversations рџ’«\n*For 18+ only*"
        else:
            text += "Now our conversations will be more restrained and romantic рџ’–"

    bot.send_message(message.chat.id, text, parse_mode='Markdown')


@bot.message_handler(commands=['profile'])
def profile_command(message):
    """РРЅС„РѕСЂРјР°С†РёСЏ Рѕ РїСЂРѕС„РёР»Рµ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ"""
    user_id = message.from_user.id
    lang = user_languages[user_id]

    user_data = user_db.get_user(user_id)
    if not user_data:
        user_db.create_user(user_id, message.from_user.username,
                            message.from_user.first_name, message.from_user.last_name)
        user_data = user_db.get_user(user_id)

    if lang == 'ru':
        premium_status = "вњ… РђРљРўРР’Р•Рќ" if user_data['is_premium'] else "вќЊ РќР•РђРљРўРР’Р•Рќ"
        voice_uses = user_db.get_voice_uses_left(user_id)

        text = (
            f"рџ‘¤ *РџР РћР¤РР›Р¬ РџРћР›Р¬Р—РћР’РђРўР•Р›РЇ*\n\n"
            f"рџ†” ID: {user_id}\n"
            f"рџ‘¤ РРјСЏ: {user_data['first_name'] or 'РќРµ СѓРєР°Р·Р°РЅРѕ'}\n"
            f"рџЊђ Username: @{user_data['username'] or 'РќРµ СѓРєР°Р·Р°РЅ'}\n\n"
            f"рџ’« *РЎРўРђРўРЈРЎ:*\n"
            f"вЂў РџСЂРµРјРёСѓРј: {premium_status}\n"
            "вЂў РќР°РІРёРіР°С†РёСЏ - /help\n"
            "вЂў РћС„РѕСЂРјРёС‚СЊ РїСЂРµРјРёСѓРј - /premium\n"
        )

        # РџРѕРєР°Р·С‹РІР°РµРј РѕС‚РєСЂРѕРІРµРЅРЅС‹Р№ СЂРµР¶РёРј С‚РѕР»СЊРєРѕ РґР»СЏ РїСЂРµРјРёСѓРј РїРѕР»СЊР·РѕРІР°С‚РµР»РµР№
        if user_data['is_premium']:
            explicit_status = "Р’РљР›Р®Р§Р•Рќ" if user_data['explicit_mode'] else "Р’Р«РљР›Р®Р§Р•Рќ"
            text += f"вЂў РћС‚РєСЂРѕРІРµРЅРЅС‹Р№ СЂРµР¶РёРј: {explicit_status}\n"

        text += f"вЂў РћСЃС‚Р°Р»РѕСЃСЊ РІРѕР№СЃРѕРІ СЃРµРіРѕРґРЅСЏ: {voice_uses}\n\n"
        text += f"рџ“… Р”Р°С‚Р° СЂРµРіРёСЃС‚СЂР°С†РёРё: {user_data['created_at'][:10] if user_data['created_at'] else 'РќРµРёР·РІРµСЃС‚РЅРѕ'}"
    else:
        premium_status = "вњ… ACTIVE" if user_data['is_premium'] else "вќЊ INACTIVE"
        voice_uses = user_db.get_voice_uses_left(user_id)

        text = (
            f"рџ‘¤ *USER PROFILE*\n\n"
            f"рџ†” ID: {user_id}\n"
            f"рџ‘¤ First name: {user_data['first_name'] or 'Not specified'}\n"
            f"рџЊђ Username: @{user_data['username'] or 'Not specified'}\n\n"
            f"рџ’« *STATUS:*\n"
            f"вЂў Premium: {premium_status}\n"
        )

        # РџРѕРєР°Р·С‹РІР°РµРј РѕС‚РєСЂРѕРІРµРЅРЅС‹Р№ СЂРµР¶РёРј С‚РѕР»СЊРєРѕ РґР»СЏ РїСЂРµРјРёСѓРј РїРѕР»СЊР·РѕРІР°С‚РµР»РµР№
        if user_data['is_premium']:
            explicit_status = "ENABLED" if user_data['explicit_mode'] else "DISABLED"
            text += f"вЂў Explicit mode: {explicit_status}\n"

        text += f"вЂў Voice messages left today: {voice_uses}\n\n"
        text += f"рџ“… Registration date: {user_data['created_at'][:10] if user_data['created_at'] else 'Unknown'}"

    bot.send_message(message.chat.id, text, parse_mode='Markdown')


@bot.message_handler(commands=['voice'])
def voice_command(message):
    """Р’РєР»СЋС‡РµРЅРёРµ/РІС‹РєР»СЋС‡РµРЅРёРµ РіРѕР»РѕСЃРѕРІС‹С… СЃРѕРѕР±С‰РµРЅРёР№"""
    user_id = message.from_user.id
    lang = user_languages[user_id]

    user_voice_enabled[user_id] = not user_voice_enabled[user_id]
    new_status = user_voice_enabled[user_id]

    if lang == 'ru':
        status = "Р’РљР›Р®Р§Р•РќР«" if new_status else "Р’Р«РљР›Р®Р§Р•РќР«"
        text = f"рџ”Љ Р“РѕР»РѕСЃРѕРІС‹Рµ СЃРѕРѕР±С‰РµРЅРёСЏ: *{status}*"
    else:
        status = "ENABLED" if new_status else "DISABLED"
        text = f"рџ”Љ Voice messages: *{status}*"

    bot.send_message(message.chat.id, text, parse_mode='Markdown')


@bot.message_handler(commands=['clear'])
def clear_command(message):
    """РћС‡РёСЃС‚РєР° РёСЃС‚РѕСЂРёРё РґРёР°Р»РѕРіР°"""
    user_id = message.from_user.id
    lang = user_languages[user_id]

    user_conversations[user_id] = []

    if lang == 'ru':
        text = "рџ§№ *РСЃС‚РѕСЂРёСЏ РґРёР°Р»РѕРіР° РѕС‡РёС‰РµРЅР°!*\n\nРўРµРїРµСЂСЊ СЏ РЅРµ РїРѕРјРЅСЋ РЅР°С€Рё РїСЂРµРґС‹РґСѓС‰РёРµ СЃРѕРѕР±С‰РµРЅРёСЏ."
    else:
        text = "рџ§№ *Conversation history cleared!*\n\nI no longer remember our previous messages."

    bot.send_message(message.chat.id, text, parse_mode='Markdown')


@bot.message_handler(commands=['status'])
def status_command(message):
    """РЎС‚Р°С‚СѓСЃ Р±РѕС‚Р° Рё РёРЅС„РѕСЂРјР°С†РёСЏ Рѕ СЃРёСЃС‚РµРјРµ"""
    user_id = message.from_user.id
    lang = user_languages[user_id]

    # РџРѕР»СѓС‡Р°РµРј РёРЅС„РѕСЂРјР°С†РёСЋ Рѕ СЃРёСЃС‚РµРјРµ
    total_users = len(user_conversations)
    active_conversations = sum(1 for conv in user_conversations.values() if len(conv) > 0)

    if lang == 'ru':
        text = (
            f"рџ¤– *РЎРўРђРўРЈРЎ Р‘РћРўРђ*\n\n"
            f"вЂў рџ¤– AI РјРѕРґРµР»СЊ: gemini\n"
            f"вЂў рџ‘Ґ Р’СЃРµРіРѕ РїРѕР»СЊР·РѕРІР°С‚РµР»РµР№: {total_users}\n"
            f"вЂў рџ’¬ РђРєС‚РёРІРЅС‹С… РґРёР°Р»РѕРіРѕРІ: {active_conversations}\n"
            f"*РљРѕРјР°РЅРґС‹:*\n"
            f"/start - РЅР°С‡Р°С‚СЊ РѕР±С‰РµРЅРёРµ\n"
            f"/profile - РёРЅС„РѕСЂРјР°С†РёСЏ Рѕ РїСЂРѕС„РёР»Рµ\n"
            f"/premium - РїСЂРµРјРёСѓРј РїРѕРґРїРёСЃРєР°\n"
            f"/voice - РІРєР»/РІС‹РєР» РіРѕР»РѕСЃРѕРІС‹Рµ\n"
            f"/clear - РѕС‡РёСЃС‚РёС‚СЊ РёСЃС‚РѕСЂРёСЋ\n"
            f"/status - СЌС‚РѕС‚ СЃС‚Р°С‚СѓСЃ"
        )
    else:
        text = (
            f"рџ¤– *BOT STATUS*\n\n"
            f"вЂў рџ¤– AI model: {ai_bot.model_name}\n"
            f"вЂў рџЋ™пёЏ Voice engine: {'Silero TTS + gTTS' if ai_bot.silero_available else 'Enhanced TTS'}\n"
            f"вЂў рџ‘Ґ Total users: {total_users}\n"
            f"вЂў рџ’¬ Active conversations: {active_conversations}\n"
            f"вЂў рџ—„пёЏ Database: users.db\n\n"
            f"*Commands:*\n"
            f"/start - start communication\n"
            f"/profile - profile information\n"
            f"/premium - premium subscription\n"
            f"/voice - enable/disable voice\n"
            f"/clear - clear history\n"
            f"/status - this status"
        )

    bot.send_message(message.chat.id, text, parse_mode='Markdown')


@bot.message_handler(commands=['help'])
def help_command(message):
    """РЎРїСЂР°РІРєР° РїРѕ РєРѕРјР°РЅРґР°Рј"""
    user_id = message.from_user.id
    lang = user_languages[user_id]

    if lang == 'ru':
        text = (
            f"рџ¤– *РџРћРњРћР©Р¬ РџРћ РљРћРњРђРќР”РђРњ*\n\n"
            f"*РћСЃРЅРѕРІРЅС‹Рµ РєРѕРјР°РЅРґС‹:*\n"
            f"/start - РЅР°С‡Р°С‚СЊ РѕР±С‰РµРЅРёРµ СЃ Р±РѕС‚РѕРј\n"
            f"/profile - РёРЅС„РѕСЂРјР°С†РёСЏ Рѕ РІР°С€РµРј РїСЂРѕС„РёР»Рµ\n"
            f"/premium - РїСЂРµРјРёСѓРј РїРѕРґРїРёСЃРєР°\n"
            f"/explicit - СѓРїСЂР°РІР»РµРЅРёРµ РѕС‚РєСЂРѕРІРµРЅРЅС‹Рј СЂРµР¶РёРјРѕРј\n"
            f"/voice - РІРєР»СЋС‡РёС‚СЊ/РІС‹РєР»СЋС‡РёС‚СЊ РіРѕР»РѕСЃРѕРІС‹Рµ СЃРѕРѕР±С‰РµРЅРёСЏ\n"
            f"/clear - РѕС‡РёСЃС‚РёС‚СЊ РёСЃС‚РѕСЂРёСЋ РґРёР°Р»РѕРіР°\n"
            f"/status - СЃС‚Р°С‚СѓСЃ Р±РѕС‚Р° Рё РёРЅС„РѕСЂРјР°С†РёСЏ Рѕ СЃРёСЃС‚РµРјРµ\n"
            f"/help - СЌС‚Р° СЃРїСЂР°РІРєР°\n\n"
            f"*РљР°Рє РїРѕР»СѓС‡РёС‚СЊ РіРѕР»РѕСЃРѕРІРѕР№ РѕС‚РІРµС‚:*\n"
            f"Р”РѕР±Р°РІСЊС‚Рµ РІ РєРѕРЅРµС† СЃРѕРѕР±С‰РµРЅРёСЏ: `СЃРєРёРЅСЊ РІРѕР№СЃ` РёР»Рё `РІРѕР№СЃ`\n\n"
            f"*Р›РёРјРёС‚С‹:*\n"
            f"вЂў Р‘РµСЃРїР»Р°С‚РЅС‹Рµ РїРѕР»СЊР·РѕРІР°С‚РµР»Рё: 3 РІРѕР№СЃР° РІ РґРµРЅСЊ\n"
            f"вЂў РџСЂРµРјРёСѓРј РїРѕР»СЊР·РѕРІР°С‚РµР»Рё: Р±РµР·Р»РёРјРёС‚РЅС‹Рµ РІРѕР№СЃС‹"
        )
    else:
        text = (
            f"рџ¤– *COMMAND HELP*\n\n"
            f"*Basic commands:*\n"
            f"/start - start communication with the bot\n"
            f"/profile - information about your profile\n"
            f"/premium - premium subscription\n"
            f"/explicit - manage explicit mode\n"
            f"/voice - enable/disable voice messages\n"
            f"/clear - clear conversation history\n"
            f"/status - bot status and system information\n"
            f"/help - this help\n\n"
            f"*How to get voice response:*\n"
            f"Add to the end of the message: `send voice` or `voice`\n\n"
            f"*Limits:*\n"
            f"вЂў Free users: 3 voice messages per day\n"
            f"вЂў Premium users: unlimited voice messages"
        )

    bot.send_message(message.chat.id, text, parse_mode='Markdown')


# РћР±СЂР°Р±РѕС‚С‡РёРєРё callback-Р·Р°РїСЂРѕСЃРѕРІ
@bot.callback_query_handler(func=lambda call: call.data == 'buy_premium_stars')
def buy_premium_stars_callback(call):
    """РћР±СЂР°Р±РѕС‚С‡РёРє РїРѕРєСѓРїРєРё РїСЂРµРјРёСѓРј РїРѕРґРїРёСЃРєРё С‡РµСЂРµР· Telegram Stars"""
    user_id = call.from_user.id
    lang = user_languages[user_id]

    try:
        # РЎРѕР·РґР°РµРј РёРЅРІРѕР№СЃ РґР»СЏ РѕРїР»Р°С‚С‹ С‡РµСЂРµР· Telegram Stars
        prices = [types.LabeledPrice(label="Premium Subscription (1 week)", amount=50)]

        # РћС‚РїСЂР°РІР»СЏРµРј РёРЅРІРѕР№СЃ
        bot.send_invoice(
            chat_id=call.message.chat.id,
            title="РџСЂРµРјРёСѓРј РїРѕРґРїРёСЃРєР° РЅР° 1 РЅРµРґРµР»СЋ" if lang == 'ru' else "Premium Subscription (1 week)",
            description="РђРєС‚РёРІР°С†РёСЏ РїСЂРµРјРёСѓРј РїРѕРґРїРёСЃРєРё РЅР° 1 РЅРµРґРµР»СЋ. Р’РєР»СЋС‡Р°РµС‚ Р±РµР·Р»РёРјРёС‚РЅС‹Рµ РІРѕР№СЃС‹, СЂР°СЃС€РёСЂРµРЅРЅСѓСЋ РїР°РјСЏС‚СЊ Рё РѕС‚РєСЂРѕРІРµРЅРЅС‹Рµ С‚РµРјС‹ 18+" if lang == 'ru' else "Premium subscription for 1 week. Includes unlimited voice messages, enhanced memory and 18+ explicit topics",
            invoice_payload=f"premium_{user_id}",
            provider_token="",
            currency="XTR",
            prices=prices,
            start_parameter="premium_subscription",
            need_name=False,
            need_phone_number=False,
            need_email=False,
            need_shipping_address=False,
            is_flexible=False
        )

        if lang == 'ru':
            bot.answer_callback_query(call.id, "рџ’° РћС‚РєСЂС‹РІР°СЋ РѕРєРЅРѕ РѕРїР»Р°С‚С‹...")
        else:
            bot.answer_callback_query(call.id, "рџ’° Opening payment window...")

    except Exception as e:
        logger.error(f"РћС€РёР±РєР° РїСЂРё СЃРѕР·РґР°РЅРёРё РёРЅРІРѕР№СЃР°: {e}")
        if lang == 'ru':
            bot.answer_callback_query(call.id, "вќЊ РћС€РёР±РєР° РїСЂРё СЃРѕР·РґР°РЅРёРё РїР»Р°С‚РµР¶Р°")
        else:
            bot.answer_callback_query(call.id, "вќЊ Error creating payment")


@bot.callback_query_handler(func=lambda call: call.data == 'buy_premium_crypto')
def buy_premium_crypto_callback(call):
    """РћР±СЂР°Р±РѕС‚С‡РёРє РїРѕРєСѓРїРєРё РїСЂРµРјРёСѓРј РїРѕРґРїРёСЃРєРё С‡РµСЂРµР· CryptoBot"""
    user_id = call.from_user.id
    lang = user_languages[user_id]

    # РџСЂРѕРІРµСЂСЏРµРј, СѓСЃС‚Р°РЅРѕРІР»РµРЅ Р»Рё С‚РѕРєРµРЅ CryptoBot
    if CRYPTO_BOT_TOKEN == "Р’РђРЁ_CRYPTOBOT_API_РўРћРљР•Рќ":
        if lang == 'ru':
            bot.answer_callback_query(call.id, "вќЊ РћРїР»Р°С‚Р° С‡РµСЂРµР· CryptoBot РІСЂРµРјРµРЅРЅРѕ РЅРµРґРѕСЃС‚СѓРїРЅР°")
            bot.send_message(call.message.chat.id,
                             "вљ пёЏ РћРїР»Р°С‚Р° С‡РµСЂРµР· CryptoBot РІСЂРµРјРµРЅРЅРѕ РЅРµРґРѕСЃС‚СѓРїРЅР°. РџРѕР¶Р°Р»СѓР№СЃС‚Р°, РёСЃРїРѕР»СЊР·СѓР№С‚Рµ РѕРїР»Р°С‚Сѓ С‡РµСЂРµР· Telegram Stars.")
        else:
            bot.answer_callback_query(call.id, "вќЊ CryptoBot payment temporarily unavailable")
            bot.send_message(call.message.chat.id,
                             "вљ пёЏ CryptoBot payment is temporarily unavailable. Please use Telegram Stars.")
        return

    try:
        # РЎРѕР·РґР°РµРј РёРЅРІРѕР№СЃ С‡РµСЂРµР· CryptoBot
        invoice = create_crypto_invoice(1.0, "USDT")

        if invoice and invoice.get('pay_url'):
            pay_url = invoice['pay_url']
            invoice_id = invoice['invoice_id']

            if lang == 'ru':
                text = (
                    f"рџ’і *РћРїР»Р°С‚Р° С‡РµСЂРµР· CryptoBot*\n\n"
                    f"РЎСѓРјРјР°: *5 USDT*\n"
                    f"РЎСЂРѕРє: *1 РЅРµРґРµР»СЏ*\n\n"
                    f"Р”Р»СЏ РѕРїР»Р°С‚С‹ РїРµСЂРµР№РґРёС‚Рµ РїРѕ СЃСЃС‹Р»РєРµ РЅРёР¶Рµ Рё СЃР»РµРґСѓР№С‚Рµ РёРЅСЃС‚СЂСѓРєС†РёСЏРј.\n"
                    f"РџРѕСЃР»Рµ РѕРїР»Р°С‚С‹ РЅР°Р¶РјРёС‚Рµ РєРЅРѕРїРєСѓ 'РџСЂРѕРІРµСЂРёС‚СЊ РѕРїР»Р°С‚Сѓ'.\n\n"
                    f"*Р’РєР»СЋС‡РµРЅРѕ:*\n"
                    f"вЂў в™ѕпёЏ Р‘РµР·Р»РёРјРёС‚РЅС‹Рµ РІРѕР№СЃС‹\n"
                    f"вЂў рџ§  РЈРІРµР»РёС‡РµРЅРЅР°СЏ РїР°РјСЏС‚СЊ\n"
                    f"вЂў рџ”ћ РћС‚РєСЂРѕРІРµРЅРЅС‹Рµ С‚РµРјС‹ 18+\n\n"
                    f"вљ пёЏ *РўРѕР»СЊРєРѕ РґР»СЏ РїРѕР»СЊР·РѕРІР°С‚РµР»РµР№ 18+*"
                )
            else:
                text = (
                    f"рџ’і *Payment via CryptoBot*\n\n"
                    f"Amount: *5 USDT*\n"
                    f"Duration: *1 week*\n\n"
                    f"To pay, follow the link below and follow the instructions.\n"
                    f"After payment, click the 'Check Payment' button.\n\n"
                    f"*Includes:*\n"
                    f"вЂў в™ѕпёЏ Unlimited voice messages\n"
                    f"вЂў рџ§  Enhanced memory\n"
                    f"вЂў рџ”ћ 18+ explicit topics\n\n"
                    f"вљ пёЏ *For users 18+ only*"
                )

            keyboard = types.InlineKeyboardMarkup()
            keyboard.add(types.InlineKeyboardButton("рџ”— РџРµСЂРµР№С‚Рё Рє РѕРїР»Р°С‚Рµ", url=pay_url))
            keyboard.add(types.InlineKeyboardButton("вњ… РџСЂРѕРІРµСЂРёС‚СЊ РѕРїР»Р°С‚Сѓ", callback_data=f'check_crypto_{invoice_id}'))

            bot.send_message(call.message.chat.id, text, parse_mode='Markdown', reply_markup=keyboard)

            if lang == 'ru':
                bot.answer_callback_query(call.id, "рџ’° РЎРѕР·РґР°РµРј РїР»Р°С‚РµР¶...")
            else:
                bot.answer_callback_query(call.id, "рџ’° Creating payment...")

        else:
            logger.error(f"РќРµ СѓРґР°Р»РѕСЃСЊ СЃРѕР·РґР°С‚СЊ РёРЅРІРѕР№СЃ CryptoBot: {invoice}")
            if lang == 'ru':
                bot.answer_callback_query(call.id, "вќЊ РћС€РёР±РєР° РїСЂРё СЃРѕР·РґР°РЅРёРё РїР»Р°С‚РµР¶Р°")
                bot.send_message(call.message.chat.id,
                                 "вљ пёЏ РќРµ СѓРґР°Р»РѕСЃСЊ СЃРѕР·РґР°С‚СЊ РїР»Р°С‚РµР¶. РџРѕР¶Р°Р»СѓР№СЃС‚Р°, РїРѕРїСЂРѕР±СѓР№С‚Рµ РїРѕР·Р¶Рµ РёР»Рё РёСЃРїРѕР»СЊР·СѓР№С‚Рµ РѕРїР»Р°С‚Сѓ С‡РµСЂРµР· Telegram Stars.")
            else:
                bot.answer_callback_query(call.id, "вќЊ Error creating payment")
                bot.send_message(call.message.chat.id,
                                 "вљ пёЏ Failed to create payment. Please try again later or use Telegram Stars.")

    except Exception as e:
        logger.error(f"РћС€РёР±РєР° РїСЂРё СЃРѕР·РґР°РЅРёРё CryptoBot РёРЅРІРѕР№СЃР°: {e}")
        if lang == 'ru':
            bot.answer_callback_query(call.id, "вќЊ РћС€РёР±РєР° РїСЂРё СЃРѕР·РґР°РЅРёРё РїР»Р°С‚РµР¶Р°")
        else:
            bot.answer_callback_query(call.id, "вќЊ Error creating payment")


@bot.callback_query_handler(func=lambda call: call.data.startswith('check_crypto_'))
def check_crypto_payment_callback(call):
    """РџСЂРѕРІРµСЂРєР° РѕРїР»Р°С‚С‹ С‡РµСЂРµР· CryptoBot"""
    user_id = call.from_user.id
    lang = user_languages[user_id]
    invoice_id = int(call.data.split('_')[2])

    try:
        is_paid = check_crypto_payment(invoice_id)

        if is_paid:
            # РђРєС‚РёРІРёСЂСѓРµРј РїСЂРµРјРёСѓРј РїРѕРґРїРёСЃРєСѓ
            user_db.activate_premium(user_id, days=7)

            if lang == 'ru':
                success_text = (
                    f"рџЋ‰ *РћРџР›РђРўРђ РџРћР”РўР’Р•Р Р–Р”Р•РќРђ!* рџЋ‰\n\n"
                    f"Р’С‹ СѓСЃРїРµС€РЅРѕ Р°РєС‚РёРІРёСЂРѕРІР°Р»Рё *РџР Р•РњРРЈРњ РџРћР”РџРРЎРљРЈ* РЅР° 1 РЅРµРґРµР»СЋ!\n\n"
                    f"*РўРµРїРµСЂСЊ РІР°Рј РґРѕСЃС‚СѓРїРЅРѕ:*\n"
                    f"вЂў в™ѕпёЏ Р‘РµР·Р»РёРјРёС‚РЅС‹Рµ РіРѕР»РѕСЃРѕРІС‹Рµ СЃРѕРѕР±С‰РµРЅРёСЏ\n"
                    f"вЂў рџ§  РЈРІРµР»РёС‡РµРЅРЅР°СЏ РїР°РјСЏС‚СЊ РґРёР°Р»РѕРіР°\n"
                    f"вЂў рџ”ћ РћС‚РєСЂРѕРІРµРЅРЅС‹Рµ С‚РµРјС‹ РґР»СЏ РІР·СЂРѕСЃР»С‹С… 18+\n"
                    f"вЂў рџ’¬ Р‘РѕР»РµРµ РіР»СѓР±РѕРєРёРµ Рё РёРЅС‚РёРјРЅС‹Рµ Р±РµСЃРµРґС‹\n\n"
                    f"РСЃРїРѕР»СЊР·СѓР№С‚Рµ /explicit С‡С‚РѕР±С‹ СѓРїСЂР°РІР»СЏС‚СЊ РѕС‚РєСЂРѕРІРµРЅРЅС‹Рј СЂРµР¶РёРјРѕРј\n\n"
                    f"РЎРїР°СЃРёР±Рѕ Р·Р° РїРѕРєСѓРїРєСѓ! рџ’«"
                )
            else:
                success_text = (
                    f"рџЋ‰ *PAYMENT CONFIRMED!* рџЋ‰\n\n"
                    f"You have successfully activated *PREMIUM SUBSCRIPTION* for 1 week!\n\n"
                    f"*Now you have access to:*\n"
                    f"вЂў в™ѕпёЏ Unlimited voice messages\n"
                    f"вЂў рџ§  Enhanced chat memory\n"
                    f"вЂў рџ”ћ 18+ explicit topics\n"
                    f"вЂў рџ’¬ Deeper and more intimate conversations\n\n"
                    f"Use /explicit to manage explicit mode\n\n"
                    f"Thank you for your purchase! рџ’«"
                )

            bot.edit_message_text(
                success_text,
                call.message.chat.id,
                call.message.message_id,
                parse_mode='Markdown'
            )

        else:
            if lang == 'ru':
                bot.answer_callback_query(call.id, "вќЊ РћРїР»Р°С‚Р° РЅРµ РЅР°Р№РґРµРЅР°. РџРѕРїСЂРѕР±СѓР№С‚Рµ РїРѕР·Р¶Рµ.")
            else:
                bot.answer_callback_query(call.id, "вќЊ Payment not found. Try again later.")

    except Exception as e:
        logger.error(f"РћС€РёР±РєР° РїСЂРё РїСЂРѕРІРµСЂРєРµ CryptoBot РїР»Р°С‚РµР¶Р°: {e}")
        if lang == 'ru':
            bot.answer_callback_query(call.id, "вќЊ РћС€РёР±РєР° РїСЂРё РїСЂРѕРІРµСЂРєРµ РїР»Р°С‚РµР¶Р°")
        else:
            bot.answer_callback_query(call.id, "вќЊ Error checking payment")


@bot.callback_query_handler(func=lambda call: call.data.startswith('lang_'))
def language_callback(call):
    """РћР±СЂР°Р±РѕС‚С‡РёРє РІС‹Р±РѕСЂР° СЏР·С‹РєР°"""
    user_id = call.from_user.id
    lang = call.data.split('_')[1]
    user_languages[user_id] = lang

    if lang == 'ru':
        welcome_text = (
            f"РџСЂРёРІРµС‚! РЇ Р§Р°СЂР»Рё - С‚РІРѕСЏ РІРёСЂС‚СѓР°Р»СЊРЅР°СЏ РїРѕРґСЂСѓРіР° рџ¤—\n\n"
            f"РЇ Р±СѓРґСѓ СЃ С‚РѕР±РѕР№ РѕР±С‰Р°С‚СЊСЃСЏ, РїРѕРґРґРµСЂР¶РёРІР°С‚СЊ Р±РµСЃРµРґСѓ Рё РѕС‚РІРµС‡Р°С‚СЊ "
            f"РіРѕР»РѕСЃРѕРІС‹РјРё СЃРѕРѕР±С‰РµРЅРёСЏРјРё!\n\n"
            f"*Р§С‚РѕР±С‹ РїРѕР»СѓС‡РёС‚СЊ РіРѕР»РѕСЃРѕРІРѕР№ РѕС‚РІРµС‚, РґРѕР±Р°РІСЊ РІ РєРѕРЅРµС† СЃРѕРѕР±С‰РµРЅРёСЏ:*\n"
            f"`СЃРєРёРЅСЊ РІРѕР№СЃ` РёР»Рё ` РІРѕР№СЃ`\n\n"
            f"*РћРіСЂР°РЅРёС‡РµРЅРёСЏ:*\n"
            f"вЂў Р‘РµСЃРїР»Р°С‚РЅС‹Рµ РїРѕР»СЊР·РѕРІР°С‚РµР»Рё: 3 РІРѕР№СЃР° РІ РґРµРЅСЊ\n"
            f"вЂў РџСЂРµРјРёСѓРј РїРѕР»СЊР·РѕРІР°С‚РµР»Рё: Р±РµР·Р»РёРјРёС‚РЅС‹Рµ РІРѕР№СЃС‹\n\n"
            f"рџ’« *РџСЂРµРјРёСѓРј РїРѕРґРїРёСЃРєР°:* /premium - 50 Stars РёР»Рё CryptoBot\n\n"
            f"Р Р°СЃСЃРєР°Р¶Рё РјРЅРµ Рѕ СЃРµР±Рµ, РїРѕРґРµР»РёСЃСЊ РјС‹СЃР»СЏРјРё РёР»Рё РїСЂРѕСЃС‚Рѕ РїРѕР·РґРѕСЂРѕРІР°Р№СЃСЏ!\n\n"
            f"*Р”РѕСЃС‚СѓРїРЅС‹Рµ РєРѕРјР°РЅРґС‹:*\n"
            f"/profile - РёРЅС„РѕСЂРјР°С†РёСЏ Рѕ РІР°С€РµРј Р°РєРєР°СѓРЅС‚Рµ\n"
            f"/premium - РїСЂРµРјРёСѓРј РїРѕРґРїРёСЃРєР°\n"
            f"/explicit - СѓРїСЂР°РІР»РµРЅРёРµ РѕС‚РєСЂРѕРІРµРЅРЅС‹Рј СЂРµР¶РёРјРѕРј\n"
            f"/voice - РІРєР»/РІС‹РєР» РіРѕР»РѕСЃРѕРІС‹Рµ СЃРѕРѕР±С‰РµРЅРёСЏ\n"
            f"/status - СЃС‚Р°С‚СѓСЃ Р±РѕС‚Р°\n"
            f"/clear - РѕС‡РёСЃС‚РёС‚СЊ РёСЃС‚РѕСЂРёСЋ РґРёР°Р»РѕРіР°\n"
            f"/help - СЃРїСЂР°РІРєР° РїРѕ РєРѕРјР°РЅРґР°Рј"
        )
    else:
        welcome_text = (
            f"Hello! I'm Charlie - your virtual girlfriend рџ¤—\n\n"
            f"рџ¤– *AI used:* {ai_bot.model_name}\n\n"
            f"рџЋ™пёЏ *Voice engine:* {'Silero TTS + gTTS' if ai_bot.silero_available else 'Enhanced TTS'}\n\n"
            f"I'll chat with you and sometimes respond with voice messages!\n\n"
            f"*Limitations:*\n"
            f"вЂў Free users: 3 voice messages per day\n"
            f"вЂў Premium users: unlimited voice messages\n\n"
            f"рџ’« *Premium subscription:* /premium - 50 Stars or CryptoBot\n\n"
            f"*To get a voice response, add to the end of your message:*\n"
            f"`send voice` or `voice message`\n\n"
            f"Tell me about yourself, share your thoughts, or just say hello!\n\n"
            f"*Available commands:*\n"
            f"/profile - information about your account\n"
            f"/premium - premium subscription\n"
            f"/explicit - manage explicit mode\n"
            f"/voice - enable/disable voice messages\n"
            f"/status - bot status\n"
            f"/clear - clear conversation history\n"
            f"/help - command help"
        )

    bot.edit_message_text(
        welcome_text,
        call.message.chat.id,
        call.message.message_id,
        parse_mode='Markdown'
    )


@bot.pre_checkout_query_handler(func=lambda query: True)
def pre_checkout_handler(pre_checkout_query):
    """РћР±СЂР°Р±РѕС‚С‡РёРє РїСЂРµРґРІР°СЂРёС‚РµР»СЊРЅРѕР№ РїСЂРѕРІРµСЂРєРё РїР»Р°С‚РµР¶Р°"""
    user_id = pre_checkout_query.from_user.id
    payload = pre_checkout_query.invoice_payload

    try:
        # РџСЂРѕРІРµСЂСЏРµРј, С‡С‚Рѕ СЌС‚Рѕ РїР»Р°С‚РµР¶ Р·Р° РїСЂРµРјРёСѓРј
        if payload.startswith('premium_'):
            # РџРѕРґС‚РІРµСЂР¶РґР°РµРј РІРѕР·РјРѕР¶РЅРѕСЃС‚СЊ РїСЂРёРЅСЏС‚СЊ РїР»Р°С‚РµР¶
            bot.answer_pre_checkout_query(pre_checkout_query.id, ok=True)
        else:
            bot.answer_pre_checkout_query(pre_checkout_query.id, ok=False,
                                          error_message="РќРµРёР·РІРµСЃС‚РЅС‹Р№ С‚РёРї РїР»Р°С‚РµР¶Р°")
    except Exception as e:
        logger.error(f"РћС€РёР±РєР° РІ pre-checkout: {e}")
        bot.answer_pre_checkout_query(pre_checkout_query.id, ok=False,
                                      error_message="РћС€РёР±РєР° РѕР±СЂР°Р±РѕС‚РєРё РїР»Р°С‚РµР¶Р°")


@bot.message_handler(content_types=['successful_payment'])
def successful_payment_handler(message):
    """РћР±СЂР°Р±РѕС‚С‡РёРє СѓСЃРїРµС€РЅРѕРіРѕ РїР»Р°С‚РµР¶Р°"""
    user_id = message.from_user.id
    payment_info = message.successful_payment
    lang = user_languages[user_id]

    try:
        # РђРєС‚РёРІРёСЂСѓРµРј РїСЂРµРјРёСѓРј РїРѕРґРїРёСЃРєСѓ
        user_db.activate_premium(user_id, days=7)

        if lang == 'ru':
            success_text = (
                f"рџЋ‰ *РћРџР›РђРўРђ РџРћР”РўР’Р•Р Р–Р”Р•РќРђ!* рџЋ‰\n\n"
                f"Р’С‹ СѓСЃРїРµС€РЅРѕ Р°РєС‚РёРІРёСЂРѕРІР°Р»Рё *РџР Р•РњРРЈРњ РџРћР”РџРРЎРљРЈ* РЅР° 1 РЅРµРґРµР»СЋ!\n\n"
                f"*РўРµРїРµСЂСЊ РІР°Рј РґРѕСЃС‚СѓРїРЅРѕ:*\n"
                f"вЂў в™ѕпёЏ Р‘РµР·Р»РёРјРёС‚РЅС‹Рµ РіРѕР»РѕСЃРѕРІС‹Рµ СЃРѕРѕР±С‰РµРЅРёСЏ\n"
                f"вЂў рџ§  РЈРІРµР»РёС‡РµРЅРЅР°СЏ РїР°РјСЏС‚СЊ РґРёР°Р»РѕРіР°\n"
                f"вЂў рџ”ћ РћС‚РєСЂРѕРІРµРЅРЅС‹Рµ С‚РµРјС‹ РґР»СЏ РІР·СЂРѕСЃР»С‹С… 18+\n"
                f"вЂў рџ’¬ Р‘РѕР»РµРµ РіР»СѓР±РѕРєРёРµ Рё РёРЅС‚РёРјРЅС‹Рµ Р±РµСЃРµРґС‹\n\n"
                f"РСЃРїРѕР»СЊР·СѓР№С‚Рµ /explicit С‡С‚РѕР±С‹ СѓРїСЂР°РІР»СЏС‚СЊ РѕС‚РєСЂРѕРІРµРЅРЅС‹Рј СЂРµР¶РёРјРѕРј\n\n"
                f"РЎРїР°СЃРёР±Рѕ Р·Р° РїРѕРєСѓРїРєСѓ! рџ’«"
            )
        else:
            success_text = (
                f"рџЋ‰ *PAYMENT CONFIRMED!* рџЋ‰\n\n"
                f"You have successfully activated *PREMIUM SUBSCRIPTION* for 1 week!\n\n"
                f"*Now you have access to:*\n"
                f"вЂў в™ѕпёЏ Unlimited voice messages\n"
                f"вЂў рџ§  Enhanced chat memory\n"
                f"вЂў рџ”ћ 18+ explicit topics\n"
                f"вЂў рџ’¬ Deeper and more intimate conversations\n\n"
                f"Use /explicit to manage explicit mode\n\n"
                f"Thank you for your purchase! рџ’«"
            )

        bot.send_message(message.chat.id, success_text, parse_mode='Markdown')
        logger.info(f"РџРѕР»СЊР·РѕРІР°С‚РµР»СЊ {user_id} Р°РєС‚РёРІРёСЂРѕРІР°Р» РїСЂРµРјРёСѓРј С‡РµСЂРµР· Stars")

    except Exception as e:
        logger.error(f"РћС€РёР±РєР° РїСЂРё Р°РєС‚РёРІР°С†РёРё РїСЂРµРјРёСѓРјР° РїРѕСЃР»Рµ РѕРїР»Р°С‚С‹: {e}")
        if lang == 'ru':
            bot.send_message(message.chat.id, "вќЊ РџСЂРѕРёР·РѕС€Р»Р° РѕС€РёР±РєР° РїСЂРё Р°РєС‚РёРІР°С†РёРё РїСЂРµРјРёСѓРјР°. РЎРІСЏР¶РёС‚РµСЃСЊ СЃ РїРѕРґРґРµСЂР¶РєРѕР№.")
        else:
            bot.send_message(message.chat.id, "вќЊ Error activating premium. Please contact support.")


def should_send_voice_message(user_text: str, lang: str) -> tuple:
    """РћРїСЂРµРґРµР»СЏРµС‚, РЅСѓР¶РЅРѕ Р»Рё РѕС‚РїСЂР°РІР»СЏС‚СЊ РіРѕР»РѕСЃРѕРІРѕРµ СЃРѕРѕР±С‰РµРЅРёРµ"""
    text_lower = user_text.lower().strip()

    if lang == 'ru':
        patterns = [
            r'.*СЃРєРёРЅСЊ\s+РІРѕР№СЃ\s*[.!?]*$',
            r'.*РѕС‚РїСЂР°РІСЊ\s+РІРѕР№СЃ\s*[.!?]*$',
            r'.*РѕС‚РІРµС‚СЊ\s+РіРѕР»РѕСЃРѕРј\s*[.!?]*$',
            r'.*РІРѕР№СЃ\s*[.!?]*$',
            r'.*РѕР·РІСѓС‡СЊ\s*[.!?]*$'
        ]

        for pattern in patterns:
            if re.match(pattern, text_lower):
                cleaned = re.sub(r'\s*(СЃРєРёРЅСЊ|РѕС‚РїСЂР°РІСЊ)\s+РІРѕР№СЃ\s*[.!?]*$', '', user_text, flags=re.IGNORECASE)
                cleaned = re.sub(r'\s*РѕС‚РІРµС‚СЊ\s+РіРѕР»РѕСЃРѕРј\s*[.!?]*$', '', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'\s*РіРѕР»РѕСЃРѕРІРѕРµ\s*[.!?]*$', '', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'\s*РѕР·РІСѓС‡СЊ\s*[.!?]*$', '', cleaned, flags=re.IGNORECASE)
                return True, cleaned.strip()

    else:
        patterns = [
            r'.*send\s+voice\s*[.!?]*$',
            r'.*send\s+voice\s+message\s*[.!?]*$',
            r'.*respond\s+with\s+voice\s*[.!?]*$',
            r'.*voice\s+message\s*[.!?]*$',
            r'.*voice\s*[.!?]*$'
        ]

        for pattern in patterns:
            if re.match(pattern, text_lower):
                cleaned = re.sub(r'\s*send\s+voice(\s+message)?\s*[.!?]*$', '', user_text, flags=re.IGNORECASE)
                cleaned = re.sub(r'\s*respond\s+with\s+voice\s*[.!?]*$', '', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'\s*voice\s+message\s*[.!?]*$', '', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'\s*voice\s*[.!?]*$', '', cleaned, flags=re.IGNORECASE)
                return True, cleaned.strip()

    return False, user_text


def send_voice_message(chat_id: int, audio_file: str, user_id: int) -> bool:
    """РћС‚РїСЂР°РІР»СЏРµС‚ РіРѕР»РѕСЃРѕРІРѕРµ СЃРѕРѕР±С‰РµРЅРёРµ РІ С„РѕСЂРјР°С‚Рµ MP3"""
    try:
        with open(audio_file, 'rb') as voice_file:
            bot.send_audio(chat_id, voice_file, title="Р“РѕР»РѕСЃРѕРІРѕРµ СЃРѕРѕР±С‰РµРЅРёРµ")
        logger.info("Р“РѕР»РѕСЃРѕРІРѕРµ СЃРѕРѕР±С‰РµРЅРёРµ СѓСЃРїРµС€РЅРѕ РѕС‚РїСЂР°РІР»РµРЅРѕ")
        chat_voice_support[chat_id] = True
        return True
    except Exception as e:
        error_msg = str(e)
        logger.error(f"РћС€РёР±РєР° РѕС‚РїСЂР°РІРєРё РіРѕР»РѕСЃРѕРІРѕРіРѕ СЃРѕРѕР±С‰РµРЅРёСЏ: {error_msg}")

        if "Voice_messages_forbidden" in error_msg or "voice messages are forbidden" in error_msg.lower():
            chat_voice_support[chat_id] = False
            logger.info(f"Р“РѕР»РѕСЃРѕРІС‹Рµ СЃРѕРѕР±С‰РµРЅРёСЏ Р·Р°РїСЂРµС‰РµРЅС‹ РІ С‡Р°С‚Рµ {chat_id}")
        else:
            user_voice_enabled[user_id] = False

        return False


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """РћР±СЂР°Р±РѕС‚С‡РёРє РІСЃРµС… С‚РµРєСЃС‚РѕРІС‹С… СЃРѕРѕР±С‰РµРЅРёР№"""
    # РџСЂРѕРїСѓСЃРєР°РµРј РєРѕРјР°РЅРґС‹ - РѕРЅРё СѓР¶Рµ РѕР±СЂР°Р±РѕС‚Р°РЅС‹ СЃРѕРѕС‚РІРµС‚СЃС‚РІСѓСЋС‰РёРјРё РѕР±СЂР°Р±РѕС‚С‡РёРєР°РјРё
    if message.text and message.text.startswith('/'):
        return

    user_id = message.from_user.id
    chat_id = message.chat.id
    user_text = message.text
    lang = user_languages[user_id]

    logger.info(f"РџРѕР»СѓС‡РµРЅРѕ СЃРѕРѕР±С‰РµРЅРёРµ РѕС‚ {user_id}: {user_text}")

    # РЈР±РµРґРёРјСЃСЏ, С‡С‚Рѕ РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РµСЃС‚СЊ РІ Р±Р°Р·Рµ
    if not user_db.get_user(user_id):
        user_db.create_user(user_id, message.from_user.username,
                            message.from_user.first_name, message.from_user.last_name)

    # РџСЂРѕРІРµСЂСЏРµРј, Р·Р°РїСЂРѕС€РµРЅРѕ Р»Рё РіРѕР»РѕСЃРѕРІРѕРµ СЃРѕРѕР±С‰РµРЅРёРµ
    send_voice, cleaned_text = should_send_voice_message(user_text, lang)

    # РџСЂРѕРІРµСЂСЏРµРј СѓСЃР»РѕРІРёСЏ РґР»СЏ РѕС‚РїСЂР°РІРєРё РіРѕР»РѕСЃРѕРІРѕРіРѕ
    can_send_voice = (
            user_voice_enabled[user_id] and
            chat_voice_support[chat_id] and
            send_voice and
            user_db.can_use_voice(user_id)
    )

    # Р•СЃР»Рё С‚РµРєСЃС‚ РїСѓСЃС‚РѕР№ РїРѕСЃР»Рµ СѓРґР°Р»РµРЅРёСЏ С‚СЂРёРіРіРµСЂР°
    if not cleaned_text.strip():
        cleaned_text = "РџСЂРёРІРµС‚" if lang == 'ru' else "Hello"

    # РџРѕРєР°Р·С‹РІР°РµРј РёРЅРґРёРєР°С‚РѕСЂ РЅР°Р±РѕСЂР°
    bot.send_chat_action(chat_id, 'typing')

    # РџРѕР»СѓС‡Р°РµРј РѕС‚РІРµС‚ РѕС‚ AI СЃ СѓС‡РµС‚РѕРј С‚РѕРіРѕ, Р·Р°РїСЂРѕС€РµРЅРѕ Р»Рё РіРѕР»РѕСЃРѕРІРѕРµ
    bot_response = ai_bot.get_ai_response(user_id, cleaned_text, lang, voice_requested=send_voice)

    # РћС‚РїСЂР°РІР»СЏРµРј РіРѕР»РѕСЃРѕРІРѕРµ РµСЃР»Рё РЅСѓР¶РЅРѕ
    if can_send_voice:
        audio_file = ai_bot.text_to_speech(bot_response, user_id, lang)
        if audio_file:
            try:
                voice_success = send_voice_message(chat_id, audio_file, user_id)

                if voice_success:
                    # РЈРІРµР»РёС‡РёРІР°РµРј СЃС‡РµС‚С‡РёРє РёСЃРїРѕР»СЊР·РѕРІР°РЅРёР№ РІРѕР№СЃРѕРІ
                    user_db.increment_voice_use(user_id)

                    # РџРѕРєР°Р·С‹РІР°РµРј СЃРєРѕР»СЊРєРѕ РІРѕР№СЃРѕРІ РѕСЃС‚Р°Р»РѕСЃСЊ
                    user_data = user_db.get_user(user_id)
                    if not user_data['is_premium']:
                        uses_left = 3 - user_data['voice_uses_today']
                        if uses_left > 0:
                            if lang == 'ru':
                                reminder = f"в„№пёЏ РћСЃС‚Р°Р»РѕСЃСЊ РІРѕР№СЃРѕРІ СЃРµРіРѕРґРЅСЏ: {uses_left}/3\nрџ’« Р‘РµР·Р»РёРјРёС‚РЅС‹Рµ РІРѕР№СЃС‹ СЃ /premium"
                            else:
                                reminder = f"в„№пёЏ Voice messages left today: {uses_left}/3\nрџ’« Unlimited voice with /premium"
                            bot.send_message(chat_id, reminder)

                if not voice_success:
                    if chat_voice_support[chat_id]:
                        if lang == 'ru':
                            bot.send_message(chat_id, "вљ пёЏ РќРµ СѓРґР°Р»РѕСЃСЊ РѕС‚РїСЂР°РІРёС‚СЊ РіРѕР»РѕСЃРѕРІРѕРµ СЃРѕРѕР±С‰РµРЅРёРµ.")
                        else:
                            bot.send_message(chat_id, "вљ пёЏ Couldn't send voice message.")
                    else:
                        if lang == 'ru':
                            bot.send_message(chat_id, "в„№пёЏ Р’ СЌС‚РѕРј С‡Р°С‚Рµ РіРѕР»РѕСЃРѕРІС‹Рµ СЃРѕРѕР±С‰РµРЅРёСЏ Р·Р°РїСЂРµС‰РµРЅС‹.")
                        else:
                            bot.send_message(chat_id, "в„№пёЏ Voice messages are forbidden in this chat.")

            except Exception as e:
                logger.error(f"РћС€РёР±РєР° РїСЂРё РѕР±СЂР°Р±РѕС‚РєРµ РіРѕР»РѕСЃРѕРІРѕРіРѕ СЃРѕРѕР±С‰РµРЅРёСЏ: {e}")
                # Р’ СЃР»СѓС‡Р°Рµ РѕС€РёР±РєРё РѕС‚РїСЂР°РІР»СЏРµРј С‚РµРєСЃС‚РѕРІС‹Р№ РѕС‚РІРµС‚
                bot.send_message(chat_id, bot_response)
            finally:
                if os.path.exists(audio_file):
                    try:
                        os.remove(audio_file)
                    except Exception as e:
                        logger.error(f"РћС€РёР±РєР° РїСЂРё СѓРґР°Р»РµРЅРёРё С„Р°Р№Р»Р°: {e}")
        else:
            # Р•СЃР»Рё РЅРµ СѓРґР°Р»РѕСЃСЊ СЃРѕР·РґР°С‚СЊ РіРѕР»РѕСЃРѕРІРѕРµ, РѕС‚РїСЂР°РІР»СЏРµРј С‚РµРєСЃС‚РѕРІС‹Р№ РѕС‚РІРµС‚
            bot.send_message(chat_id, bot_response)
    else:
        # Р•СЃР»Рё РіРѕР»РѕСЃРѕРІРѕРµ РЅРµ Р·Р°РїСЂРѕС€РµРЅРѕ РёР»Рё РЅРµРґРѕСЃС‚СѓРїРЅРѕ, РѕС‚РїСЂР°РІР»СЏРµРј С‚РµРєСЃС‚РѕРІС‹Р№ РѕС‚РІРµС‚
        bot.send_message(chat_id, bot_response)

        # Р•СЃР»Рё Р·Р°РїСЂРѕС€РµРЅРѕ РіРѕР»РѕСЃРѕРІРѕРµ, РЅРѕ РїСЂРµРІС‹С€РµРЅ Р»РёРјРёС‚
        if send_voice and not user_db.can_use_voice(user_id):
            user_data = user_db.get_user(user_id)
            if not user_data['is_premium']:
                if lang == 'ru':
                    bot.send_message(
                        chat_id,
                        f"вќЊ Р›РёРјРёС‚ РІРѕР№СЃРѕРІ РёСЃС‡РµСЂРїР°РЅ!3/3 РІРѕР№СЃРѕРІ СЃРµРіРѕРґРЅСЏ.\n\n"
                        f"рџ’« РџСЂРµРјРёСѓРј РїРѕР»СЊР·РѕРІР°С‚РµР»Рё РёРјРµСЋС‚ Р±РµР·Р»РёРјРёС‚РЅС‹Рµ РІРѕР№СЃС‹!\n"
                        f"РСЃРїРѕР»СЊР·СѓР№С‚Рµ /premium РґР»СЏ Р°РєС‚РёРІР°С†РёРё Р·Р° 50 Р·РІРµР·Рґ"
                    )
                else:
                    bot.send_message(
                        chat_id,
                        f"вќЊ Voice message limit reached! You've used 3/3 voice messages today.\n\n"
                        f"рџ’« *Premium users* get unlimited voice messages!\n"
                        f"Use /premium to activate for 50 Telegram Stars"
                    )

        if send_voice and not chat_voice_support[chat_id]:
            if lang == 'ru':
                bot.send_message(chat_id, "в„№пёЏ Р’ СЌС‚РѕРј С‡Р°С‚Рµ РіРѕР»РѕСЃРѕРІС‹Рµ СЃРѕРѕР±С‰РµРЅРёСЏ Р·Р°РїСЂРµС‰РµРЅС‹.")
            else:
                bot.send_message(chat_id, "в„№пёЏ Voice messages are forbidden in this chat.")


if __name__ == '__main__':
    # РџСЂРѕРІРµСЂСЏРµРј РєРѕРЅС„РёРіСѓСЂР°С†РёСЋ РїРµСЂРµРґ Р·Р°РїСѓСЃРєРѕРј
    validate_config()

    print("=" * 50)
    print("рџ¤– Р‘РѕС‚ РЁР°СЂР»РѕС‚С‚Р° Р·Р°РїСѓСЃРєР°РµС‚СЃСЏ...")
    print(f"рџ¤– РСЃРїРѕР»СЊР·СѓРµРјС‹Р№ AI: {ai_bot.model_name}")
    print(f"рџЋ™пёЏ Р“РѕР»РѕСЃРѕРІРѕР№ РґРІРёР¶РѕРє: {'Silero TTS + gTTS' if ai_bot.silero_available else 'РЈР»СѓС‡С€РµРЅРЅС‹Р№ TTS'}")
    print(f"рџ’ѕ Р‘Р°Р·Р° РґР°РЅРЅС‹С…: users.db")
    print(f"рџ’« РЎРёСЃС‚РµРјР° РѕРїР»Р°С‚С‹: Telegram Stars + CryptoBot")
    print(f"рџ”ћ РџСЂРµРјРёСѓРј СЂРµР¶РёРј: СѓРїСЂР°РІР»РµРЅРёРµ РѕС‚РєСЂРѕРІРµРЅРЅС‹РјРё С‚РµРјР°РјРё")
    print("=" * 50)

    try:
        bot.infinity_polling()
    except Exception as e:
        logger.error(f"РљСЂРёС‚РёС‡РµСЃРєР°СЏ РѕС€РёР±РєР° Р±РѕС‚Р°: {e}")
        print(f"РљСЂРёС‚РёС‡РµСЃРєР°СЏ РѕС€РёР±РєР°: {e}")
