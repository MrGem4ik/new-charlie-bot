#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import io
import os

# ЖЕСТКАЯ НАСТРОЙКА КОДИРОВКИ
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'

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
import sqlite3
import json
import threading

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') 
CRYPTO_BOT_TOKEN = os.environ.get('CRYPTO_BOT_TOKEN')

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, threaded=True)

user_conversations = defaultdict(lambda: [])
user_languages = defaultdict(lambda: 'ru')
user_voice_enabled = defaultdict(lambda: True)
chat_voice_support = defaultdict(lambda: True)

class UserDatabase:
    def __init__(self, db_path='users.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_database()

    def init_database(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
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
            
            # Включаем WAL режим для лучшей производительности
            cursor.execute('PRAGMA journal_mode=WAL')
            cursor.execute('PRAGMA synchronous=NORMAL')
            
            try:
                cursor.execute("SELECT explicit_mode FROM users LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute('ALTER TABLE users ADD COLUMN explicit_mode BOOLEAN DEFAULT FALSE')
            conn.commit()
            conn.close()
            logger.info("База данных инициализирована")

    def get_user(self, user_id):
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute('SELECT user_id, username, first_name, last_name, is_premium, premium_until, stars, voice_uses_today, last_voice_date, explicit_mode, created_at FROM users WHERE user_id = ?', (user_id,))
            user = cursor.fetchone()
            conn.close()
            if user:
                return {
                    'user_id': user[0], 'username': user[1], 'first_name': user[2], 'last_name': user[3],
                    'is_premium': bool(user[4]), 'premium_until': user[5], 'stars': user[6],
                    'voice_uses_today': user[7], 'last_voice_date': user[8], 'explicit_mode': bool(user[9]),
                    'created_at': user[10]
                }
            return None

    def create_user(self, user_id, username, first_name, last_name):
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM users WHERE user_id = ?', (user_id,))
            exists = cursor.fetchone()
            if not exists:
                cursor.execute('INSERT INTO users (user_id, username, first_name, last_name, stars, explicit_mode) VALUES (?, ?, ?, ?, ?, ?)', (user_id, username, first_name, last_name, 0, False))
            else:
                cursor.execute('UPDATE users SET username = ?, first_name = ?, last_name = ? WHERE user_id = ?', (username, first_name, last_name, user_id))
            conn.commit()
            conn.close()

    def update_stars(self, user_id, stars):
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET stars = ? WHERE user_id = ?', (stars, user_id))
            conn.commit()
            conn.close()

    def activate_premium(self, user_id, days=7):
        with self.lock:
            premium_until = datetime.now() + timedelta(days=days)
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET is_premium = TRUE, premium_until = ? WHERE user_id = ?', (premium_until.strftime('%Y-%m-%d'), user_id))
            conn.commit()
            conn.close()

    def toggle_explicit_mode(self, user_id):
        with self.lock:
            user = self.get_user(user_id)
            if user and user['is_premium']:
                new_mode = not user['explicit_mode']
                conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
                cursor = conn.cursor()
                cursor.execute('UPDATE users SET explicit_mode = ? WHERE user_id = ?', (new_mode, user_id))
                conn.commit()
                conn.close()
                return new_mode
            return False

    def add_stars(self, user_id, amount):
        with self.lock:
            user = self.get_user(user_id)
            if user:
                new_stars = user['stars'] + amount
                self.update_stars(user_id, new_stars)
                return new_stars
            return 0

    def can_use_voice(self, user_id):
        with self.lock:
            user = self.get_user(user_id)
            if not user: 
                return True
            today = date.today()
            last_date = user['last_voice_date']
            if last_date != str(today):
                self.reset_voice_counter(user_id)
                return True
            if user['is_premium']: 
                return True
            else: 
                return user['voice_uses_today'] < 3

    def increment_voice_use(self, user_id):
        with self.lock:
            today = str(date.today())
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET voice_uses_today = voice_uses_today + 1, last_voice_date = ? WHERE user_id = ?', (today, user_id))
            conn.commit()
            conn.close()

    def reset_voice_counter(self, user_id):
        with self.lock:
            today = str(date.today())
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET voice_uses_today = 0, last_voice_date = ? WHERE user_id = ?', (today, user_id))
            conn.commit()
            conn.close()

    def get_voice_uses_left(self, user_id):
        with self.lock:
            user = self.get_user(user_id)
            if not user or user['is_premium']: 
                return "∞"
            return max(0, 3 - user['voice_uses_today'])

user_db = UserDatabase()

class AIChatBot:
    def __init__(self):
        self.gemini_model_standard = None
        self.gemini_model_premium = None
        self.model_name = "Локальный интеллект"
        self.silero_available = self.check_silero_availability()
        self.initialize_gemini_models()

    def check_silero_availability(self):
        try:
            import torch
            device = torch.device('cpu')
            torch.set_num_threads(4)
            model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='ru', speaker='v3_1_ru')
            logger.info("Silero TTS доступен")
            return True
        except Exception as e:
            logger.warning(f"Silero TTS недоступен: {e}")
            return False

    def initialize_gemini_models(self):
        try:
            if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
                logger.error("GEMINI_API_KEY не установлен")
                return
            genai.configure(api_key=GEMINI_API_KEY)
            try:
                models = genai.list_models()
                available_models = [model.name for model in models]
                chat_models = [model for model in available_models if any(x in model for x in ['gemini', 'gemma']) and not any(x in model for x in ['embedding', 'imagen', 'veo', 'aqa', 'learnlm'])]
            except Exception as e:
                logger.warning(f"Не удалось получить список моделей: {e}")
                chat_models = []
            priority_models = ['models/gemini-2.0-flash', 'models/gemini-2.0-flash-001', 'models/gemini-2.0-flash-lite', 'models/gemini-2.0-flash-lite-001', 'models/gemini-flash-latest', 'models/gemini-pro-latest', 'models/gemini-2.5-flash', 'models/gemma-3-27b-it', 'models/gemma-3-12b-it', 'models/gemma-3-4b-it']
            models_to_try = []
            for model in priority_models:
                if model in chat_models: 
                    models_to_try.append(model)
            if not models_to_try and chat_models: 
                models_to_try = chat_models[:5]
            if not models_to_try: 
                models_to_try = priority_models
            safety_settings_standard = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
            safety_settings_premium = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
            self.gemini_model_standard = self._initialize_model_with_settings(models_to_try, safety_settings_standard, "стандартная")
            self.gemini_model_premium = self._initialize_model_with_settings(models_to_try, safety_settings_premium, "премиум")
            if self.gemini_model_standard or self.gemini_model_premium:
                model_names = []
                if self.gemini_model_standard: 
                    model_names.append("стандартная")
                if self.gemini_model_premium: 
                    model_names.append("премиум")
                self.model_name = f"Gemini: {', '.join(model_names)}"
            else: 
                logger.error("Все модели Gemini недоступны")
        except Exception as e: 
            logger.error(f"Ошибка инициализации Gemini: {str(e)}")

    def _initialize_model_with_settings(self, models_to_try, safety_settings, model_type):
        for model_name in models_to_try:
            try:
                generation_config = {"temperature": 0.9, "top_p": 0.95, "top_k": 40, "max_output_tokens": 200}
                model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config, safety_settings=safety_settings)
                test_response = model.generate_content("Привет! Ответь коротко: как дела?")
                if test_response and test_response.text:
                    logger.info(f"Успешно инициализирована {model_type} модель: {model_name}")
                    return model
            except Exception as e:
                error_str = str(e)
                if "quota" in error_str.lower() or "429" in error_str: 
                    break
                continue
        return None

    def get_ai_response(self, user_id: int, user_text: str, lang: str = 'ru', voice_requested: bool = False) -> str:
        user_data = user_db.get_user(user_id)
        is_premium = user_data['is_premium'] if user_data else False
        explicit_mode = user_data['explicit_mode'] if user_data and is_premium else False
        if is_premium and explicit_mode and self.gemini_model_premium:
            gemini_response = self.try_gemini_response(user_id, user_text, lang, voice_requested, is_premium, explicit_mode, self.gemini_model_premium)
        else:
            gemini_response = self.try_gemini_response(user_id, user_text, lang, voice_requested, is_premium, explicit_mode, self.gemini_model_standard)
        if gemini_response: 
            return gemini_response
        return self.get_enhanced_fallback_response(user_id, user_text, lang, voice_requested)

    def try_gemini_response(self, user_id: int, user_text: str, lang: str, voice_requested: bool = False, is_premium: bool = False, explicit_mode: bool = False, gemini_model=None) -> str:
        if not gemini_model: 
            return None
        try:
            history = user_conversations[user_id]
            max_history_length = 30 if is_premium else 12
            if lang == 'ru':
                if voice_requested:
                    base_prompt = """Ты Чарли, милая, умная студентка. Отвечай естественно, кратко (1-2 предложения). Будь игривой, немного кокетливой. ОБЯЗАТЕЛЬНО используй правильные знаки препинания. НЕ используй смайлики и эмодзи в ответе!"""
                else:
                    base_prompt = """Ты Чарли, милая, озорная студентка. Отвечай естественно, кратко (1-2 предложения). Будь игривой, немного кокетливой. Можно использовать смайлики."""
            else:
                if voice_requested:
                    base_prompt = """You are Charlie. Respond naturally, briefly (1-2 sentences). Be playful, a bit flirty. ALWAYS use proper punctuation. DO NOT use emojis in your response!"""
                else:
                    base_prompt = """You are Charlie. Respond naturally, briefly (1-2 sentences). Be playful, a bit flirty. You can use emojis moderately."""
            if is_premium and explicit_mode:
                if lang == 'ru': 
                    premium_prompt = "\n[ПРЕМИУМ РЕЖИМ 18+]\nТеперь ты можешь обсуждать более интимные темы."
                else: 
                    premium_prompt = "\n[PREMIUM MODE 18+]\nNow you can discuss more intimate topics."
                system_prompt = base_prompt + premium_prompt
            else: 
                system_prompt = base_prompt
            conversation_history = ""
            for msg in history[-max_history_length:]:
                if msg["role"] == "user": 
                    conversation_history += f"Пользователь: {msg['content']}\n"
                else: 
                    conversation_history += f"Ты: {msg['content']}\n"
            prompt = f"{system_prompt}\n\nИстория диалога:\n{conversation_history}\nПользователь: {user_text}\n\nТвой ответ:"
            response = gemini_model.generate_content(prompt)
            if response and response.text:
                bot_response = response.text.strip()
                bot_response = self.clean_response(bot_response, voice_requested)
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": bot_response})
                if len(history) > max_history_length: 
                    user_conversations[user_id] = history[-max_history_length:]
                return bot_response
        except Exception as e:
            logger.error(f"Ошибка при запросе к Gemini: {str(e)}")
        return None

    def clean_response(self, response: str, voice_requested: bool = False) -> str:
        if not response: 
            return "Интересно! Расскажи больше." if not voice_requested else "Интересно, расскажи больше."
        response = response.replace('*', '').replace('**', '').strip()
        if response.startswith('Ты:') or response.startswith('You:'): 
            response = response.split(':', 1)[1].strip()
        if voice_requested: 
            response = self.remove_emojis(response)
        if response and not response.endswith(('.', '!', '?')): 
            response += '.'
        if len(response) < 2: 
            return "Расскажи мне больше об этом!" if not voice_requested else "Расскажи мне больше об этом."
        return response

    def get_enhanced_fallback_response(self, user_id: int, user_text: str, lang: str, voice_requested: bool = False) -> str:
        user_text_lower = user_text.lower()
        history = user_conversations[user_id]
        user_data = user_db.get_user(user_id)
        is_premium = user_data['is_premium'] if user_data else False
        max_history_length = 30 if is_premium else 12
        if lang == 'ru':
            if any(word in user_text_lower for word in ['привет', 'здравств', 'добрый', 'hi', 'hello', 'хай', 'ку']): 
                responses = ["Привет! Рада тебя видеть! 😊"]
            elif any(word in user_text_lower for word in ['как дела', 'как ты', 'настроен']): 
                responses = ["Всё прекрасно! А у тебя как? 😉"]
            else: 
                responses = ["Расскажи мне больше об этом! 💫"]
        else: 
            responses = ["Tell me more about it! 💫"]
        bot_response = random.choice(responses)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": bot_response})
        if len(history) > max_history_length: 
            user_conversations[user_id] = history[-max_history_length:]
        return bot_response

    def preprocess_text_for_speech(self, text: str) -> str:
        emoji_replacements = {
            '))': ', улыбаясь,', 
            ')))': ', смеясь,', 
            ':)': ', улыбаясь,', 
            ':(': ', с грустью,', 
            ';)': ', подмигивая,', 
            '<3': ', с любовью,'
        }
        for emoji, replacement in emoji_replacements.items(): 
            text = text.replace(emoji, replacement)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def text_to_speech(self, text: str, user_id: int, lang: str = 'ru') -> str:
        try:
            processed_text = self.preprocess_text_for_speech(text)
            processed_text = self.remove_emojis(processed_text)
            if len(processed_text) > 1000: 
                processed_text = processed_text[:1000] + "..."
            audio_filename = f"voice_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            
            # Пробуем gTTS сначала
            try:
                if lang == 'ru': 
                    tts = gTTS(text=processed_text, lang='ru', slow=False, lang_check=False)
                else: 
                    tts = gTTS(text=processed_text, lang='en', slow=False, lang_check=False)
                tts.save(audio_filename)
                if os.path.exists(audio_filename): 
                    return audio_filename
            except Exception as e:
                logger.warning(f"gTTS error: {e}")
            
            return None
        except Exception as e: 
            logger.error(f"TTS error: {e}")
            return None

    def remove_emojis(self, text: str) -> str:
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

ai_bot = AIChatBot()

def create_crypto_invoice(amount: float, currency: str = "USDT") -> dict:
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
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get("ok"): 
                return data.get("result")
        return None
    except Exception as e: 
        return None

def check_crypto_payment(invoice_id: int) -> bool:
    try:
        url = "https://pay.crypt.bot/api/getInvoices"
        params = {"invoice_ids": str(invoice_id), "status": "paid"}
        headers = {"Crypto-Pay-API-Token": CRYPTO_BOT_TOKEN}
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get("ok") and data.get("result", {}).get("items"): 
                return len(data["result"]["items"]) > 0
        return False
    except Exception as e: 
        return False

@bot.message_handler(commands=['start'])
def start_command(message):
    user_id = message.from_user.id
    if not user_db.get_user(user_id): 
        user_db.create_user(user_id, message.from_user.username, message.from_user.first_name, message.from_user.last_name)
    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(types.InlineKeyboardButton("🇷🇺 Русский", callback_data='lang_ru'))
    keyboard.add(types.InlineKeyboardButton("🇺🇸 English", callback_data='lang_en'))
    bot.send_message(message.chat.id, "Please choose your language / Пожалуйста, выберите язык:", reply_markup=keyboard)

@bot.message_handler(commands=['premium'])
def premium_command(message):
    user_id = message.from_user.id
    lang = user_languages[user_id]
    user_data = user_db.get_user(user_id)
    if not user_data: 
        user_db.create_user(user_id, message.from_user.username, message.from_user.first_name, message.from_user.last_name)
        user_data = user_db.get_user(user_id)
    is_premium = user_data['is_premium']
    if lang == 'ru':
        if is_premium:
            premium_until = user_data['premium_until']
            explicit_status = "ВКЛЮЧЕН" if user_data['explicit_mode'] else "ВЫКЛЮЧЕН"
            text = f"🔓 *ПРЕМИУМ СТАТУС*\n\n✅ У вас активна премиум подписка!\n📅 Действует до: {premium_until}\n🔞 Откровенный режим: {explicit_status}\n\n*Преимущества:*\n• ♾️ Безлимитные войсы\n• 🧠 Увеличенная память\n• 🔞 Откровенные темы: {explicit_status}\n• 💬 Более глубокие беседы\n\nИспользуйте /explicit чтобы переключить откровенный режим"
        else:
            text = f"🔓 *ПРЕМИУМ ПОДПИСКА*\n\nПолучите эксклюзивные возможности на неделю!\n\n*🔓 ВКЛЮЧАЕТ:*\n• ♾️ Безлимитные войсы\n• 🧠 Увеличенная память\n• 🔞 Откровенные темы 18+\n• 💬 Более глубокие беседы\n\n*💳 СПОСОБЫ ОПЛАТЫ:*\n• 50 Telegram Stars\n• 💰 CryptoBot\n\n*⚠️ Откровенные темы только для 18+*"
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            keyboard.add(
                types.InlineKeyboardButton("💫 50 Stars", callback_data='buy_premium_stars'), 
                types.InlineKeyboardButton("💰 CryptoBot", callback_data='buy_premium_crypto')
            )
    else:
        if is_premium:
            premium_until = user_data['premium_until']
            explicit_status = "ENABLED" if user_data['explicit_mode'] else "DISABLED"
            text = f"🔓 *PREMIUM STATUS*\n\n✅ You have an active premium subscription!\n📅 Valid until: {premium_until}\n🔞 Explicit mode: {explicit_status}\n\n*Benefits:*\n• ♾️ Unlimited voice messages\n• 🧠 Enhanced memory\n• 🔞 18+ explicit topics\n• 💬 Deeper conversations\n\nUse /explicit to toggle explicit mode"
        else:
            text = f"🔓 *PREMIUM SUBSCRIPTION*\n\nGet exclusive features for 1 week!\n\n*🔓 INCLUDES:*\n• ♾️ Unlimited voice messages\n• 🧠 Enhanced memory\n• 🔞 18+ explicit topics\n• 💬 Deeper conversations\n\n*💳 PAYMENT METHODS:*\n• 50 Telegram Stars\n• 💰 CryptoBot\n\n*⚠️ Explicit topics for 18+ only*"
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            keyboard.add(
                types.InlineKeyboardButton("💫 50 Stars", callback_data='buy_premium_stars'), 
                types.InlineKeyboardButton("💰 CryptoBot", callback_data='buy_premium_crypto')
            )
    if is_premium: 
        bot.send_message(message.chat.id, text, parse_mode='Markdown')
    else: 
        bot.send_message(message.chat.id, text, parse_mode='Markdown', reply_markup=keyboard)

@bot.message_handler(commands=['explicit'])
def explicit_command(message):
    user_id = message.from_user.id
    lang = user_languages[user_id]
    user_data = user_db.get_user(user_id)
    if not user_data: 
        user_db.create_user(user_id, message.from_user.username, message.from_user.first_name, message.from_user.last_name)
        user_data = user_db.get_user(user_id)
    if not user_data['is_premium']:
        if lang == 'ru': 
            bot.send_message(message.chat.id, "❌ Эта функция доступна только для премиум пользователей!")
        else: 
            bot.send_message(message.chat.id, "❌ This feature is available only for premium users!")
        return
    new_mode = user_db.toggle_explicit_mode(user_id)
    if lang == 'ru':
        status = "ВКЛЮЧЕН" if new_mode else "ВЫКЛЮЧЕН"
        text = f"🔞 Режим откровенных тем: *{status}*\n\n"
        if new_mode: 
            text += "Теперь я готова к более откровенным беседам 💫\n*Только для 18+*"
        else: 
            text += "Теперь наши беседы будут более сдержанными 💖"
    else:
        status = "ENABLED" if new_mode else "DISABLED"
        text = f"🔞 Explicit mode: *{status}*\n\n"
        if new_mode: 
            text += "Now I'm ready for more open conversations 💫\n*For 18+ only*"
        else: 
            text += "Now our conversations will be more restrained 💖"
    bot.send_message(message.chat.id, text, parse_mode='Markdown')

@bot.message_handler(commands=['profile'])
def profile_command(message):
    user_id = message.from_user.id
    lang = user_languages[user_id]
    user_data = user_db.get_user(user_id)
    if not user_data: 
        user_db.create_user(user_id, message.from_user.username, message.from_user.first_name, message.from_user.last_name)
        user_data = user_db.get_user(user_id)
    if lang == 'ru':
        premium_status = "✅ АКТИВЕН" if user_data['is_premium'] else "❌ НЕАКТИВЕН"
        voice_uses = user_db.get_voice_uses_left(user_id)
        text = f"👤 *ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ*\n\n🆔 ID: {user_id}\n👤 Имя: {user_data['first_name'] or 'Не указано'}\n📎 Username: @{user_data['username'] or 'Не указан'}\n\n💫 *СТАТУС:*\n• Премиум: {premium_status}\n"
        if user_data['is_premium']: 
            text += f"• Откровенный режим: {'ВКЛЮЧЕН' if user_data['explicit_mode'] else 'ВЫКЛЮЧЕН'}\n"
        text += f"• Осталось войсов сегодня: {voice_uses}\n\n📅 Дата регистрации: {user_data['created_at'][:10] if user_data['created_at'] else 'Неизвестно'}"
    else:
        premium_status = "✅ ACTIVE" if user_data['is_premium'] else "❌ INACTIVE"
        voice_uses = user_db.get_voice_uses_left(user_id)
        text = f"👤 *USER PROFILE*\n\n🆔 ID: {user_id}\n👤 First name: {user_data['first_name'] or 'Not specified'}\n📎 Username: @{user_data['username'] or 'Not specified'}\n\n💫 *STATUS:*\n• Premium: {premium_status}\n"
        if user_data['is_premium']: 
            text += f"• Explicit mode: {'ENABLED' if user_data['explicit_mode'] else 'DISABLED'}\n"
        text += f"• Voice messages left today: {voice_uses}\n\n📅 Registration date: {user_data['created_at'][:10] if user_data['created_at'] else 'Unknown'}"
    bot.send_message(message.chat.id, text, parse_mode='Markdown')

@bot.message_handler(commands=['voice'])
def voice_command(message):
    user_id = message.from_user.id
    lang = user_languages[user_id]
    user_voice_enabled[user_id] = not user_voice_enabled[user_id]
    new_status = user_voice_enabled[user_id]
    if lang == 'ru': 
        text = f"🔉 Голосовые сообщения: *{'ВКЛЮЧЕНЫ' if new_status else 'ВЫКЛЮЧЕНЫ'}*"
    else: 
        text = f"🔉 Voice messages: *{'ENABLED' if new_status else 'DISABLED'}*"
    bot.send_message(message.chat.id, text, parse_mode='Markdown')

@bot.message_handler(commands=['clear'])
def clear_command(message):
    user_id = message.from_user.id
    lang = user_languages[user_id]
    user_conversations[user_id] = []
    if lang == 'ru': 
        text = "🧹 *История диалога очищена!*\n\nТеперь я не помню наши предыдущие сообщения."
    else: 
        text = "🧹 *Conversation history cleared!*\n\nI no longer remember our previous messages."
    bot.send_message(message.chat.id, text, parse_mode='Markdown')

@bot.message_handler(commands=['status'])
def status_command(message):
    user_id = message.from_user.id
    lang = user_languages[user_id]
    total_users = len(user_conversations)
    active_conversations = sum(1 for conv in user_conversations.values() if len(conv) > 0)
    if lang == 'ru': 
        text = f"🤖 *СТАТУС БОТА*\n\n• 🤖 AI модель: gemini\n• 👥 Всего пользователей: {total_users}\n• 💬 Активных диалогов: {active_conversations}\n*Команды:*\n/start - начать общение\n/profile - информация о профиле\n/premium - премиум подписка\n/voice - вкл/выкл голосовые\n/clear - очистить историю\n/status - этот статус"
    else: 
        text = f"🤖 *BOT STATUS*\n\n• 🤖 AI model: {ai_bot.model_name}\n• 👥 Total users: {total_users}\n• 💬 Active conversations: {active_conversations}\n*Commands:*\n/start - start communication\n/profile - profile information\n/premium - premium subscription\n/voice - enable/disable voice\n/clear - clear history\n/status - this status"
    bot.send_message(message.chat.id, text, parse_mode='Markdown')

@bot.message_handler(commands=['help'])
def help_command(message):
    user_id = message.from_user.id
    lang = user_languages[user_id]
    if lang == 'ru': 
        text = f"🤖 *ПОМОЩЬ ПО КОМАНДАМ*\n\n*Основные команды:*\n/start - начать общение с ботом\n/profile - информация о вашем профиле\n/premium - премиум подписка\n/explicit - управление откровенным режимом\n/voice - включить/выключить голосовые сообщения\n/clear - очистить историю диалога\n/status - статус бота и информация о системе\n/help - эта справка\n\n*Как получить голосовой ответ:*\nДобавьте в конец сообщения: `скинь войс` или `войс`\n\n*Лимиты:*\n• Бесплатные пользователи: 3 войса в день\n• Премиум пользователи: безлимитные войсы"
    else: 
        text = f"🤖 *COMMAND HELP*\n\n*Basic commands:*\n/start - start communication with the bot\n/profile - information about your profile\n/premium - premium subscription\n/explicit - manage explicit mode\n/voice - enable/disable voice messages\n/clear - clear conversation history\n/status - bot status and system information\n/help - this help\n\n*How to get voice response:*\nAdd to the end of the message: `send voice` or `voice`\n\n*Limits:*\n• Free users: 3 voice messages per day\n• Premium users: unlimited voice messages"
    bot.send_message(message.chat.id, text, parse_mode='Markdown')

@bot.callback_query_handler(func=lambda call: call.data == 'buy_premium_stars')
def buy_premium_stars_callback(call):
    user_id = call.from_user.id
    lang = user_languages[user_id]
    try:
        prices = [types.LabeledPrice(label="Premium Subscription (1 week)", amount=50)]
        bot.send_invoice(
            chat_id=call.message.chat.id,
            title="Премиум подписка на 1 неделю" if lang == 'ru' else "Premium Subscription (1 week)",
            description="Активация премиум подписки на 1 неделю." if lang == 'ru' else "Premium subscription for 1 week.",
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
            bot.answer_callback_query(call.id, "💳 Открываю окно оплаты...")
        else: 
            bot.answer_callback_query(call.id, "💳 Opening payment window...")
    except Exception as e:
        if lang == 'ru': 
            bot.answer_callback_query(call.id, "❌ Ошибка при создании платежа")
        else: 
            bot.answer_callback_query(call.id, "❌ Error creating payment")

@bot.callback_query_handler(func=lambda call: call.data == 'buy_premium_crypto')
def buy_premium_crypto_callback(call):
    user_id = call.from_user.id
    lang = user_languages[user_id]
    if CRYPTO_BOT_TOKEN == "ВАШ_CRYPTOBOT_API_ТОКЕН":
        if lang == 'ru': 
            bot.answer_callback_query(call.id, "❌ Оплата через CryptoBot временно недоступна")
        else: 
            bot.answer_callback_query(call.id, "❌ CryptoBot payment temporarily unavailable")
        return
    try:
        invoice = create_crypto_invoice(1.0, "USDT")
        if invoice and invoice.get('pay_url'):
            pay_url = invoice['pay_url']
            invoice_id = invoice['invoice_id']
            if lang == 'ru': 
                text = f"💳 *Оплата через CryptoBot*\n\nСумма: *5 USDT*\nСрок: *1 неделя*\n\nДля оплаты перейдите по ссылке ниже.\nПосле оплаты нажмите 'Проверить оплату'.\n\n*Включено:*\n• ♾️ Безлимитные войсы\n• 🧠 Увеличенная память\n• 🔞 Откровенные темы 18+\n\n⚠️ *Только для 18+*"
            else: 
                text = f"💳 *Payment via CryptoBot*\n\nAmount: *5 USDT*\nDuration: *1 week*\n\nTo pay, follow the link below.\nAfter payment, click 'Check Payment'.\n\n*Includes:*\n• ♾️ Unlimited voice messages\n• 🧠 Enhanced memory\n• 🔞 18+ explicit topics\n\n⚠️ *For 18+ only*"
            keyboard = types.InlineKeyboardMarkup()
            keyboard.add(types.InlineKeyboardButton("🔗 Перейти к оплате", url=pay_url))
            keyboard.add(types.InlineKeyboardButton("✅ Проверить оплату", callback_data=f'check_crypto_{invoice_id}'))
            bot.send_message(call.message.chat.id, text, parse_mode='Markdown', reply_markup=keyboard)
            if lang == 'ru': 
                bot.answer_callback_query(call.id, "💳 Создаем платеж...")
            else: 
                bot.answer_callback_query(call.id, "💳 Creating payment...")
        else:
            if lang == 'ru': 
                bot.answer_callback_query(call.id, "❌ Ошибка при создании платежа")
            else: 
                bot.answer_callback_query(call.id, "❌ Error creating payment")
    except Exception as e:
        if lang == 'ru': 
            bot.answer_callback_query(call.id, "❌ Ошибка при создании платежа")
        else: 
            bot.answer_callback_query(call.id, "❌ Error creating payment")

@bot.callback_query_handler(func=lambda call: call.data.startswith('check_crypto_'))
def check_crypto_payment_callback(call):
    user_id = call.from_user.id
    lang = user_languages[user_id]
    invoice_id = int(call.data.split('_')[2])
    try:
        is_paid = check_crypto_payment(invoice_id)
        if is_paid:
            user_db.activate_premium(user_id, days=7)
            if lang == 'ru': 
                success_text = f"🎉 *ОПЛАТА ПОДТВЕРЖДЕНА!*\n\nВы успешно активировали *ПРЕМИУМ ПОДПИСКУ* на 1 неделю!\n\n*Теперь вам доступно:*\n• ♾️ Безлимитные голосовые сообщения\n• 🧠 Увеличенная память диалога\n• 🔞 Откровенные темы для взрослых 18+\n• 💬 Более глубокие беседы\n\nИспользуйте /explicit чтобы управлять откровенным режимом\n\nСпасибо за покупку! 💫"
            else: 
                success_text = f"🎉 *PAYMENT CONFIRMED!*\n\nYou have successfully activated *PREMIUM SUBSCRIPTION* for 1 week!\n\n*Now you have access to:*\n• ♾️ Unlimited voice messages\n• 🧠 Enhanced chat memory\n• 🔞 18+ explicit topics\n• 💬 Deeper conversations\n\nUse /explicit to manage explicit mode\n\nThank you for your purchase! 💫"
            bot.edit_message_text(success_text, call.message.chat.id, call.message.message_id, parse_mode='Markdown')
        else:
            if lang == 'ru': 
                bot.answer_callback_query(call.id, "❌ Оплата не найдена. Попробуйте позже.")
            else: 
                bot.answer_callback_query(call.id, "❌ Payment not found. Try again later.")
    except Exception as e:
        if lang == 'ru': 
            bot.answer_callback_query(call.id, "❌ Ошибка при проверке платежа")
        else: 
            bot.answer_callback_query(call.id, "❌ Error checking payment")

@bot.callback_query_handler(func=lambda call: call.data.startswith('lang_'))
def language_callback(call):
    user_id = call.from_user.id
    lang = call.data.split('_')[1]
    user_languages[user_id] = lang
    if lang == 'ru': 
        welcome_text = f"Привет! Я Чарли - твоя виртуальная подруга 💗\n\nЯ буду с тобой общаться и отвечать голосовыми сообщениями!\n\n*Чтобы получить голосовой ответ, добавь в конец сообщения:*\n`скинь войс` или ` войс`\n\n*Ограничения:*\n• Бесплатные пользователи: 3 войса в день\n• Премиум пользователи: безлимитные войсы\n\n💫 *Премиум подписка:* /premium\n\nРасскажи мне о себе!\n\n*Команды:*\n/profile - информация\n/premium - премиум\n/voice - голосовые\n/status - статус\n/clear - очистить историю\n/help - справка"
    else: 
        welcome_text = f"Hello! I'm Charlie - your virtual girlfriend 💗\n\nI'll chat with you and respond with voice messages!\n\n*To get voice response, add to your message:*\n`send voice` or `voice`\n\n*Limits:*\n• Free users: 3 voice messages per day\n• Premium users: unlimited\n\n💫 *Premium subscription:* /premium\n\nTell me about yourself!\n\n*Commands:*\n/profile - information\n/premium - premium\n/voice - voice messages\n/status - status\n/clear - clear history\n/help - help"
    bot.edit_message_text(welcome_text, call.message.chat.id, call.message.message_id, parse_mode='Markdown')

@bot.pre_checkout_query_handler(func=lambda query: True)
def pre_checkout_handler(pre_checkout_query):
    user_id = pre_checkout_query.from_user.id
    payload = pre_checkout_query.invoice_payload
    try:
        if payload.startswith('premium_'): 
            bot.answer_pre_checkout_query(pre_checkout_query.id, ok=True)
        else: 
            bot.answer_pre_checkout_query(pre_checkout_query.id, ok=False, error_message="Неизвестный тип платежа")
    except Exception as e: 
        bot.answer_pre_checkout_query(pre_checkout_query.id, ok=False, error_message="Ошибка обработки платежа")

@bot.message_handler(content_types=['successful_payment'])
def successful_payment_handler(message):
    user_id = message.from_user.id
    lang = user_languages[user_id]
    try:
        user_db.activate_premium(user_id, days=7)
        if lang == 'ru': 
            success_text = f"🎉 *ОПЛАТА ПОДТВЕРЖДЕНА!*\n\nВы успешно активировали *ПРЕМИУМ ПОДПИСКУ* на 1 неделю!\n\n*Теперь вам доступно:*\n• ♾️ Безлимитные голосовые сообщения\n• 🧠 Увеличенная память диалога\n• 🔞 Откровенные темы для взрослых 18+\n• 💬 Более глубокие беседы\n\nИспользуйте /explicit чтобы управлять откровенным режимом\n\nСпасибо за покупку! 💫"
        else: 
            success_text = f"🎉 *PAYMENT CONFIRMED!*\n\nYou have successfully activated *PREMIUM SUBSCRIPTION* for 1 week!\n\n*Now you have access to:*\n• ♾️ Unlimited voice messages\n• 🧠 Enhanced chat memory\n• 🔞 18+ explicit topics\n• 💬 Deeper conversations\n\nUse /explicit to manage explicit mode\n\nThank you for your purchase! 💫"
        bot.send_message(message.chat.id, success_text, parse_mode='Markdown')
    except Exception as e:
        if lang == 'ru': 
            bot.send_message(message.chat.id, "❌ Произошла ошибка при активации премиума.")
        else: 
            bot.send_message(message.chat.id, "❌ Error activating premium.")

def should_send_voice_message(user_text: str, lang: str) -> tuple:
    text_lower = user_text.lower().strip()
    if lang == 'ru':
        patterns = [
            r'.*скинь\s+войс\s*[.!?]*$', 
            r'.*отправь\s+войс\s*[.!?]*$', 
            r'.*ответь\s+голосом\s*[.!?]*$', 
            r'.*войс\s*[.!?]*$', 
            r'.*озвучь\s*[.!?]*$'
        ]
        for pattern in patterns:
            if re.match(pattern, text_lower):
                cleaned = re.sub(r'\s*(скинь|отправь)\s+войс\s*[.!?]*$', '', user_text, flags=re.IGNORECASE)
                cleaned = re.sub(r'\s*ответь\s+голосом\s*[.!?]*$', '', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'\s*голосовое\s*[.!?]*$', '', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'\s*озвучь\s*[.!?]*$', '', cleaned, flags=re.IGNORECASE)
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
    try:
        if os.path.exists(audio_file):
            with open(audio_file, 'rb') as voice_file: 
                bot.send_audio(chat_id, voice_file, title="Голосовое сообщение")
            chat_voice_support[chat_id] = True
            return True
    except Exception as e:
        error_msg = str(e)
        if "Voice_messages_forbidden" in error_msg or "voice messages are forbidden" in error_msg.lower(): 
            chat_voice_support[chat_id] = False
        else: 
            user_voice_enabled[user_id] = False
        return False
    return False

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    if message.text and message.text.startswith('/'): 
        return
    user_id = message.from_user.id
    chat_id = message.chat.id
    user_text = message.text
    if not user_text:
        return
    lang = user_languages[user_id]
    if not user_db.get_user(user_id): 
        user_db.create_user(user_id, message.from_user.username, message.from_user.first_name, message.from_user.last_name)
    send_voice, cleaned_text = should_send_voice_message(user_text, lang)
    can_send_voice = (user_voice_enabled[user_id] and chat_voice_support[chat_id] and send_voice and user_db.can_use_voice(user_id))
    if not cleaned_text.strip(): 
        cleaned_text = "Привет" if lang == 'ru' else "Hello"
    bot.send_chat_action(chat_id, 'typing')
    bot_response = ai_bot.get_ai_response(user_id, cleaned_text, lang, voice_requested=send_voice)
    if can_send_voice:
        audio_file = ai_bot.text_to_speech(bot_response, user_id, lang)
        if audio_file:
            try:
                voice_success = send_voice_message(chat_id, audio_file, user_id)
                if voice_success:
                    user_db.increment_voice_use(user_id)
                    user_data = user_db.get_user(user_id)
                    if not user_data['is_premium']:
                        uses_left = 3 - user_data['voice_uses_today']
                        if uses_left > 0:
                            if lang == 'ru': 
                                reminder = f"🔔 Осталось войсов сегодня: {uses_left}/3\n💫 Безлимитные войсы с /premium"
                            else: 
                                reminder = f"🔔 Voice messages left today: {uses_left}/3\n💫 Unlimited voice with /premium"
                            bot.send_message(chat_id, reminder)
                if not voice_success:
                    if chat_voice_support[chat_id]:
                        if lang == 'ru': 
                            bot.send_message(chat_id, "⚠️ Не удалось отправить голосовое сообщение.")
                        else: 
                            bot.send_message(chat_id, "⚠️ Couldn't send voice message.")
                    else:
                        if lang == 'ru': 
                            bot.send_message(chat_id, "🔔 В этом чате голосовые сообщения запрещены.")
                        else: 
                            bot.send_message(chat_id, "🔔 Voice messages are forbidden in this chat.")
            except Exception as e: 
                bot.send_message(chat_id, bot_response)
            finally:
                if os.path.exists(audio_file):
                    try: 
                        os.remove(audio_file)
                    except Exception as e: 
                        pass
        else: 
            bot.send_message(chat_id, bot_response)
    else:
        bot.send_message(chat_id, bot_response)
        if send_voice and not user_db.can_use_voice(user_id):
            user_data = user_db.get_user(user_id)
            if not user_data['is_premium']:
                if lang == 'ru': 
                    bot.send_message(chat_id, f"❌ Лимит войсов исчерпан! 3/3 войсов сегодня.\n\n💫 Премиум пользователи имеют безлимитные войсы!\n/premium - активация за 50 Stars")
                else: 
                    bot.send_message(chat_id, f"❌ Voice message limit reached! 3/3 today.\n\n💫 Premium users get unlimited voice!\n/premium - activate for 50 Stars")
        if send_voice and not chat_voice_support[chat_id]:
            if lang == 'ru': 
                bot.send_message(chat_id, "🔔 В этом чате голосовые сообщения запрещены.")
            else: 
                bot.send_message(chat_id, "🔔 Voice messages are forbidden in this chat.")

if __name__ == '__main__':
    try: 
        bot.delete_webhook()
        logger.info("Вебхук удален")
    except Exception as e: 
        logger.error(f"Ошибка при удалении вебхука: {e}")
    logger.info("Бот запускается...")
    while True:
        try: 
            bot.infinity_polling(timeout=30, long_polling_timeout=20)
        except Exception as e:
            logger.error(f"Ошибка бота: {e}")
            time.sleep(10)
