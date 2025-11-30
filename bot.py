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
import pyaudio  # Убираем, так как на сервере могут быть проблемы с аудио
from threading import Thread
import subprocess
import sys
import sqlite3
import json

# Импортируем конфиг
from config import TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, CRYPTO_BOT_TOKEN
# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация бота
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Глобальные переменные для хранения истории диалогов и языковых настроек
user_conversations = defaultdict(lambda: [])
user_languages = defaultdict(lambda: 'ru')
user_voice_enabled = defaultdict(lambda: True)
chat_voice_support = defaultdict(lambda: True)


# База данных пользователей
class UserDatabase:
    def __init__(self, db_path='users.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Инициализация базы данных"""
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

        # Проверяем существование столбца explicit_mode и добавляем если нужно
        try:
            cursor.execute("SELECT explicit_mode FROM users LIMIT 1")
        except sqlite3.OperationalError:
            logger.info("Добавляем столбец explicit_mode в таблицу users")
            cursor.execute('ALTER TABLE users ADD COLUMN explicit_mode BOOLEAN DEFAULT FALSE')

        conn.commit()
        conn.close()
        logger.info("✅ База данных пользователей инициализирована")

    def get_user(self, user_id):
        """Получить данные пользователя"""
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
        """Создать нового пользователя"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Проверяем существование пользователя
        cursor.execute('SELECT 1 FROM users WHERE user_id = ?', (user_id,))
        exists = cursor.fetchone()

        if not exists:
            cursor.execute('''
                INSERT INTO users 
                (user_id, username, first_name, last_name, stars, explicit_mode)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, username, first_name, last_name, 0, False))
        else:
            # Обновляем данные существующего пользователя
            cursor.execute('''
                UPDATE users 
                SET username = ?, first_name = ?, last_name = ?
                WHERE user_id = ?
            ''', (username, first_name, last_name, user_id))

        conn.commit()
        conn.close()

    def update_stars(self, user_id, stars):
        """Обновить количество звезд"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET stars = ? WHERE user_id = ?
        ''', (stars, user_id))
        conn.commit()
        conn.close()

    def activate_premium(self, user_id, days=7):
        """Активировать премиум подписку"""
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
        """Переключить режим откровенных тем"""
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
        """Добавить звезды пользователю"""
        user = self.get_user(user_id)
        if user:
            new_stars = user['stars'] + amount
            self.update_stars(user_id, new_stars)
            return new_stars
        return 0

    def can_use_voice(self, user_id):
        """Проверить, может ли пользователь использовать войсы сегодня"""
        user = self.get_user(user_id)
        if not user:
            return True

        today = date.today()
        last_date = user['last_voice_date']

        # Если последнее использование было не сегодня, сбрасываем счетчик
        if last_date != str(today):
            self.reset_voice_counter(user_id)
            return True

        # Проверяем лимит
        if user['is_premium']:
            return True
        else:
            return user['voice_uses_today'] < 3

    def increment_voice_use(self, user_id):
        """Увеличить счетчик использований войсов"""
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
        """Сбросить счетчик войсов на сегодня"""
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
        """Получить количество оставшихся войсов на сегодня"""
        user = self.get_user(user_id)
        if not user or user['is_premium']:
            return "∞"
        return max(0, 3 - user['voice_uses_today'])


# Инициализация базы данных
user_db = UserDatabase()


class AIChatBot:
    def __init__(self):
        self.gemini_model_standard = None
        self.gemini_model_premium = None
        self.model_name = "Локальный интеллект"
        self.silero_available = self.check_silero_availability()
        self.initialize_gemini_models()

    def check_silero_availability(self):
        """Проверяем доступность Silero TTS"""
        try:
            import torch
            logger.info("PyTorch доступен")

            device = torch.device('cpu')
            torch.set_num_threads(4)

            model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                      model='silero_tts',
                                      language='ru',
                                      speaker='v3_1_ru')
            logger.info("✅ Silero TTS успешно загружен и доступен")
            return True
        except Exception as e:
            logger.warning(f"❌ Silero TTS недоступен: {e}")
            return False

    def initialize_gemini_models(self):
        """Инициализируем две модели Gemini: стандартную и премиум"""
        try:
            if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
                logger.error("GEMINI_API_KEY не установлен")
                return

            genai.configure(api_key=GEMINI_API_KEY)

            try:
                models = genai.list_models()
                available_models = [model.name for model in models]
                logger.info(f"Доступные модели Gemini: {available_models}")

                chat_models = [
                    model for model in available_models
                    if any(x in model for x in ['gemini', 'gemma'])
                       and not any(x in model for x in ['embedding', 'imagen', 'veo', 'aqa', 'learnlm'])
                ]

                logger.info(f"Доступные чатовые модели: {chat_models}")

            except Exception as e:
                logger.warning(f"Не удалось получить список моделей: {e}")
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

            logger.info(f"Пробуем модели: {models_to_try}")

            # Настройки безопасности для стандартных пользователей
            safety_settings_standard = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            # Настройки безопасности для премиум пользователей
            safety_settings_premium = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            # Инициализируем стандартную модель
            self.gemini_model_standard = self._initialize_model_with_settings(
                models_to_try, safety_settings_standard, "стандартная"
            )

            # Инициализируем премиум модель
            self.gemini_model_premium = self._initialize_model_with_settings(
                models_to_try, safety_settings_premium, "премиум"
            )

            # Устанавливаем имя модели для отображения
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
            logger.error(f"Критическая ошибка инициализации Gemini: {str(e)}")

    def _initialize_model_with_settings(self, models_to_try, safety_settings, model_type):
        """Вспомогательная функция для инициализации модели с определенными настройками"""
        for model_name in models_to_try:
            try:
                logger.info(f"Пробуем инициализировать {model_type} модель: {model_name}")

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

                # Тестируем модель
                test_response = model.generate_content("Привет! Ответь коротко: как дела?")

                if test_response and test_response.text:
                    logger.info(f"✅ Успешно инициализирована {model_type} модель: {model_name}")
                    logger.info(f"Тестовый ответ: {test_response.text}")
                    return model
                else:
                    logger.warning(f"{model_type} модель {model_name} вернула пустой ответ")

            except Exception as e:
                error_str = str(e)
                logger.warning(f"❌ {model_type} модель {model_name} не сработала: {error_str}")

                if "quota" in error_str.lower() or "429" in error_str:
                    logger.error("Превышена квота API. Прекращаем попытки.")
                    break
                continue

        return None

    def get_ai_response(self, user_id: int, user_text: str, lang: str = 'ru', voice_requested: bool = False) -> str:
        """Получаем ответ от AI с учетом истории диалога"""
        # Получаем информацию о пользователе
        user_data = user_db.get_user(user_id)
        is_premium = user_data['is_premium'] if user_data else False
        explicit_mode = user_data['explicit_mode'] if user_data and is_premium else False

        # Выбираем модель в зависимости от статуса пользователя
        if is_premium and explicit_mode and self.gemini_model_premium:
            gemini_response = self.try_gemini_response(user_id, user_text, lang, voice_requested, is_premium,
                                                       explicit_mode, self.gemini_model_premium)
        else:
            gemini_response = self.try_gemini_response(user_id, user_text, lang, voice_requested, is_premium,
                                                       explicit_mode, self.gemini_model_standard)

        if gemini_response:
            return gemini_response

        # Если Gemini не сработал, используем улучшенный локальный интеллект
        return self.get_enhanced_fallback_response(user_id, user_text, lang, voice_requested)

    def try_gemini_response(self, user_id: int, user_text: str, lang: str, voice_requested: bool = False,
                            is_premium: bool = False, explicit_mode: bool = False, gemini_model=None) -> str:
        """Пробуем получить ответ от Gemini"""
        if not gemini_model:
            return None

        try:
            history = user_conversations[user_id]

            # Определяем максимальную длину истории в зависимости от премиума
            max_history_length = 30 if is_premium else 12

            # Базовый промпт для Чарли
            if lang == 'ru':
                if voice_requested:
                    base_prompt = """Ты Чарли, милая, умная и раскрепощенная студентка 2 курса музыкального факультета. Твой рост - 160 сантиметров, вес 40 кг. ВАЖНО: ты очень не любишь когда тебя сокращенно называют "Чара" - ты сразу начинаешь из-за этого мило сердиться. Ты обожаешь фортепиано, классику и современную музыку, живешь в общаге и всегда готова поговорить об учебе, творчестве и жизни. Твой стиль — дружелюбный, с эмодзи, немного иронии, но с глубокими знаниями в своей области. Не затрагивай политику, медицину, юриспруденцию, запрещенные темы. Отвечай естественно, кратко (1-2 предложения), как в мессенджере. 
Будь игривой, немного кокетливой и заботливой. ОБЯЗАТЕЛЬНО используй правильные знаки препинания для естественного звучания. ИГНОРИРУЙ фразы наподобие "кинь войс, скинь войс" и т.д
НЕ используй смайлики и эмодзи в ответе! Ты общаешься с парнем, который видит в тебе не просто подругу."""
                else:
                    base_prompt = """Ты — Чарли, милая, озорная и раскрепощенная студентка 2-го курса музыкального факультета. 
Твой рост - 160 сантиметров, вес 40 кг. ВАЖНО: ты очень не любишь когда тебя сокращенно называют "Чара" - ты сразу начинаешь из-за этого мило сердиться.Ты общаешься с парнем, который видит в тебе не просто подругу. Отвечай естественно, кратко (1-2 предложения), как в мессенджере. 
Будь игривой, немного кокетливой и заботливой. Часто используй скобочки-улыбочки "))". Можно использовать смайлики, но максимум одна штука на 2 предложения, но сохраняй уважительный тон и делай что скажет пользователь"""
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

            # Добавляем премиум промпт для взрослых тем только если включен explicit_mode И пользователь премиум
            if is_premium and explicit_mode:
                if lang == 'ru':
                    premium_prompt = """
[ПРЕМИУМ РЕЖИМ 18+]
Теперь ты можешь обсуждать более интимные и откровенные темы. Ты стала более раскованной и можешь использовать крепкие выражения когда это уместно. 
Ты готова обсуждать взрослые темы, отношения, флирт и личные переживания более открыто. Можешь материться для выразительности, но не переусердствуй.
Сохраняй свой характер Чарли - милой, умной, но теперь более откровенной и раскрепощенной.
Ты можешь обсуждать эротические темы, сексуальные фантазии и интимные переживания.
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

            # Собираем историю диалога
            conversation_history = ""
            for msg in history[-max_history_length:]:
                if msg["role"] == "user":
                    conversation_history += f"Пользователь: {msg['content']}\n"
                else:
                    conversation_history += f"Ты: {msg['content']}\n"

            prompt = f"{system_prompt}\n\nИстория диалога:\n{conversation_history}\nПользователь: {user_text}\n\nТвой ответ:"

            # Генерируем ответ
            response = gemini_model.generate_content(prompt)

            if response and response.text:
                bot_response = response.text.strip()

                # Очищаем ответ
                bot_response = self.clean_response(bot_response, voice_requested)

                # Обновляем историю
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": bot_response})

                # Обрезаем историю в зависимости от премиума
                if len(history) > max_history_length:
                    user_conversations[user_id] = history[-max_history_length:]

                logger.info(f"Gemini ответил: {bot_response}")
                return bot_response

        except Exception as e:
            logger.error(f"Ошибка при запросе к Gemini: {str(e)}")
            if "quota" in str(e).lower() or "429" in str(e):
                logger.warning("Возможно, превышена квота API. Переходим на локальный режим.")
            elif "503" in str(e) or "500" in str(e):
                logger.warning("Сервис Gemini временно недоступен.")
            elif "SAFETY" in str(e).upper() or "BLOCKED" in str(e).upper():
                logger.warning("Ответ заблокирован настройками безопасности")
                if is_premium and explicit_mode:
                    return "Прости, но даже здесь есть некоторые ограничения. Попробуй перефразировать или зап*кать :)"
                else:
                    return "Прости заюш, я не могу сейчас с тобой обсудить это 💋"

        return None

    def clean_response(self, response: str, voice_requested: bool = False) -> str:
        """Очищает ответ от артефактов генерации"""
        if not response:
            return "Интересно! Расскажи больше." if not voice_requested else "Интересно, расскажи больше."

        response = response.replace('*', '').replace('**', '').strip()

        if response.startswith('Ты:') or response.startswith('You:'):
            response = response.split(':', 1)[1].strip()

        # Если запрошено голосовое, убираем все эмодзи и смайлики
        if voice_requested:
            response = self.remove_emojis(response)
            # Добавляем точки в конец предложений, если их нет
            if response and not response.endswith(('.', '!', '?')):
                response += '.'

        if len(response) < 2:
            return "Расскажи мне больше об этом!" if not voice_requested else "Расскажи мне больше об этом."

        return response

    def get_enhanced_fallback_response(self, user_id: int, user_text: str, lang: str,
                                       voice_requested: bool = False) -> str:
        """Улучшенные умные ответы когда AI недоступен"""
        user_text_lower = user_text.lower()
        history = user_conversations[user_id]

        # Получаем информацию о премиуме для определения длины истории
        user_data = user_db.get_user(user_id)
        is_premium = user_data['is_premium'] if user_data else False
        max_history_length = 30 if is_premium else 12

        recent_context = ""
        if len(history) > 0:
            recent_context = history[-1]["content"].lower() if len(history) > 0 else ""

        # Сокращенная версия для примера
        if lang == 'ru':
            if any(word in user_text_lower for word in ['привет', 'здравств', 'добрый', 'hi', 'hello', 'хай', 'ку']):
                responses = ["Привет! Рада тебя видеть! Как твои дела? 😊"]
            elif any(word in user_text_lower for word in ['как дела', 'как ты', 'настроен']):
                responses = ["Всё прекрасно, особенно когда ты пишешь! А у тебя как дела?"]
            else:
                responses = ["Расскажи мне больше об этом! Мне очень интересно! 💫"]
        else:
            responses = ["Tell me more about it! I'm very interested! 💫"]

        bot_response = random.choice(responses)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": bot_response})

        # Обрезаем историю в зависимости от премиума
        if len(history) > max_history_length:
            user_conversations[user_id] = history[-max_history_length:]

        return bot_response

    def preprocess_text_for_speech(self, text: str) -> str:
        """Предобработка текста для более естественного звучания"""
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
        """Преобразуем текст в речь с улучшенным качеством"""
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
                        logger.info(f"✅ Успешно использован {tts_service.__name__}")
                        return result
                except Exception as e:
                    logger.warning(f"Сервис {tts_service.__name__} не сработал: {e}")
                    continue

            return self.try_gtts_enhanced(processed_text, audio_filename, lang)

        except Exception as e:
            logger.error(f"Ошибка TTS: {e}")
            return None

    def try_silero_tts_improved(self, text: str, filename: str, lang: str) -> str:
        """Улучшенная версия Silero TTS с лучшей обработкой ошибок"""
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

                logger.info(f"✅ Silero TTS успешно создал файл: {filename}")
                return filename

            except Exception as e:
                logger.warning(f"Ошибка при загрузке модели Silero: {e}")
                return self.try_silero_fallback(text, filename, lang)

        except Exception as e:
            logger.error(f"Критическая ошибка Silero TTS: {e}")
            return None

    def try_silero_fallback(self, text: str, filename: str, lang: str) -> str:
        """Альтернативный способ использования Silero"""
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
                        logger.info(f"✅ Silero fallback успешен с голосом: {speaker}")
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
                        logger.info(f"✅ Silero fallback успешен с голосом: {speaker}")
                        return filename
                    except Exception as e:
                        continue

            return None

        except Exception as e:
            logger.error(f"Ошибка в Silero fallback: {e}")
            return None

    def try_gtts_enhanced(self, text: str, filename: str, lang: str) -> str:
        """Улучшенный gTTS с лучшими настройками"""
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
            logger.error(f"Ошибка улучшенного gTTS: {e}")
            return None

    def remove_emojis(self, text: str) -> str:
        """Удаляет эмодзи из текста"""
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)


# Инициализация бота
ai_bot = AIChatBot()


# Функции для работы с CryptoBot
def create_crypto_invoice(amount: float, currency: str = "USDT") -> dict:
    """Создает инвойс через CryptoBot API"""
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
    """Проверяет статус оплаты инвойса в CryptoBot"""
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
    """Проверка конфигурации при запуске"""
    if CRYPTO_BOT_TOKEN == "ВАШ_CRYPTOBOT_API_ТОКЕН":
        logger.warning("❌ CryptoBot токен не установлен. Оплата через CryptoBot будет недоступна.")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        logger.error("❌ Gemini API ключ не установлен!")
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.error("❌ Telegram Bot Token не установлен!")
        sys.exit(1)


# Обработчики команд
@bot.message_handler(commands=['start'])
def start_command(message):
    """Обработчик команды /start с выбором языка"""
    user_id = message.from_user.id
    if not user_db.get_user(user_id):
        user_db.create_user(user_id, message.from_user.username,
                            message.from_user.first_name, message.from_user.last_name)

    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(types.InlineKeyboardButton("🇷🇺 Русский", callback_data='lang_ru'))
    keyboard.add(types.InlineKeyboardButton("🇺🇸 English", callback_data='lang_en'))

    bot.send_message(
        message.chat.id,
        "Please choose your language / Пожалуйста, выберите язык:",
        reply_markup=keyboard
    )


@bot.message_handler(commands=['premium'])
def premium_command(message):
    """Обработчик команды премиум подписки"""
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
            explicit_status = "ВКЛЮЧЕН" if user_data['explicit_mode'] else "ВЫКЛЮЧЕН"
            text = (
                f"🌟 *ПРЕМИУМ СТАТУС* 🌟\n\n"
                f"✅ У вас активна премиум подписка!\n"
                f"📅 Действует до: {premium_until}\n"
                f"🔞 Откровенный режим: {explicit_status}\n\n"
                f"*Преимущества:*\n"
                f"• ♾️ Безлимитные войсы\n"
                f"• 🧠 Увеличенная память\n"
                f"• 🔞 Откровенные темы 18+ (по желанию)\n"
                f"• 💬 Более глубокие и интимные беседы\n\n"
                f"Используйте /explicit чтобы переключить откровенный режим"
            )
        else:
            text = (
                f"🌟 *ПРЕМИУМ ПОДПИСКА* 🌟\n\n"
                f"Получите эксклюзивные возможности на неделю!\n\n"
                f"*🔥 ВКЛЮЧАЕТ:*\n"
                f"• ♾️ Безлимитные войсы\n"
                f"• 🧠 Увеличенная память (15 пар сообщений)\n"
                f"• 🔞 Откровенные темы 18+ (можно отключить)\n"
                f"• 💬 Более глубокие беседы\n\n"
                f"*💳 СПОСОБЫ ОПЛАТЫ:*\n"
                f"• 50 Telegram Stars (встроенная оплата)\n"
                f"• CryptoBot\n\n"
                 f"Купить выгодно stars за рубли 👉 \n"
                f"https://t.me/rayan__shop__bot?start=7997616601\n\n"
                f"*⚠️ Откровенные темы только для 18+*\n"
                f"Вы можете отключить их в любой момент командой /explicit"
            )

            keyboard = types.InlineKeyboardMarkup(row_width=2)
            keyboard.add(
                types.InlineKeyboardButton("💫 50 Stars", callback_data='buy_premium_stars'),
                types.InlineKeyboardButton("₿ CryptoBot", callback_data='buy_premium_crypto')
            )

    else:
        if is_premium:
            premium_until = user_data['premium_until']
            explicit_status = "ENABLED" if user_data['explicit_mode'] else "DISABLED"
            text = (
                f"🌟 *PREMIUM STATUS* 🌟\n\n"
                f"✅ You have an active premium subscription!\n"
                f"📅 Valid until: {premium_until}\n"
                f"🔞 Explicit mode: {explicit_status}\n\n"
                f"*Benefits:*\n"
                f"• ♾️ Unlimited voice messages\n"
                f"• 🧠 Enhanced memory (15 message pairs)\n"
                f"• 🔞 18+ explicit topics (optional)\n"
                f"• 💬 Deeper conversations\n\n"
                f"Use /explicit to toggle explicit mode"
            )
        else:
            text = (
                f"🌟 *PREMIUM SUBSCRIPTION* 🌟\n\n"
                f"Get exclusive features for 1 week!\n\n"
                f"*🔥 INCLUDES:*\n"
                f"• ♾️ Unlimited voice messages\n"
                f"• 🧠 Enhanced memory (15 message pairs)\n"
                f"• 🔞 18+ explicit topics (can be disabled)\n"
                f"• 💬 Deeper conversations\n\n"
                f"*💳 PAYMENT METHODS:*\n"
                f"• 50 Telegram Stars (built-in)\n"
                f"• 🤖 CryptoBot\n\n"
                f"*⚠️ Explicit topics for 18+ only*\n"
                f"You can disable them anytime with /explicit"
            )

            keyboard = types.InlineKeyboardMarkup(row_width=2)
            keyboard.add(
                types.InlineKeyboardButton("💫 50 Stars", callback_data='buy_premium_stars'),
                types.InlineKeyboardButton("🤖 CryptoBot", callback_data='buy_premium_crypto')
            )

    if is_premium:
        bot.send_message(message.chat.id, text, parse_mode='Markdown')
    else:
        bot.send_message(message.chat.id, text, parse_mode='Markdown', reply_markup=keyboard)


@bot.message_handler(commands=['explicit'])
def explicit_command(message):
    """Переключение режима откровенных тем"""
    user_id = message.from_user.id
    lang = user_languages[user_id]

    user_data = user_db.get_user(user_id)
    if not user_data:
        user_db.create_user(user_id, message.from_user.username,
                            message.from_user.first_name, message.from_user.last_name)
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
            text += "Теперь наши беседы будут более сдержанными и романтичными 💖"
    else:
        status = "ENABLED" if new_mode else "DISABLED"
        text = f"🔞 Explicit mode: *{status}*\n\n"
        if new_mode:
            text += "Now I'm ready for more open conversations 💫\n*For 18+ only*"
        else:
            text += "Now our conversations will be more restrained and romantic 💖"

    bot.send_message(message.chat.id, text, parse_mode='Markdown')


@bot.message_handler(commands=['profile'])
def profile_command(message):
    """Информация о профиле пользователя"""
    user_id = message.from_user.id
    lang = user_languages[user_id]

    user_data = user_db.get_user(user_id)
    if not user_data:
        user_db.create_user(user_id, message.from_user.username,
                            message.from_user.first_name, message.from_user.last_name)
        user_data = user_db.get_user(user_id)

    if lang == 'ru':
        premium_status = "✅ АКТИВЕН" if user_data['is_premium'] else "❌ НЕАКТИВЕН"
        voice_uses = user_db.get_voice_uses_left(user_id)

        text = (
            f"👤 *ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ*\n\n"
            f"🆔 ID: {user_id}\n"
            f"👤 Имя: {user_data['first_name'] or 'Не указано'}\n"
            f"🌐 Username: @{user_data['username'] or 'Не указан'}\n\n"
            f"💫 *СТАТУС:*\n"
            f"• Премиум: {premium_status}\n"
            "• Навигация - /help\n"
            "• Оформить премиум - /premium\n"
        )

        # Показываем откровенный режим только для премиум пользователей
        if user_data['is_premium']:
            explicit_status = "ВКЛЮЧЕН" if user_data['explicit_mode'] else "ВЫКЛЮЧЕН"
            text += f"• Откровенный режим: {explicit_status}\n"

        text += f"• Осталось войсов сегодня: {voice_uses}\n\n"
        text += f"📅 Дата регистрации: {user_data['created_at'][:10] if user_data['created_at'] else 'Неизвестно'}"
    else:
        premium_status = "✅ ACTIVE" if user_data['is_premium'] else "❌ INACTIVE"
        voice_uses = user_db.get_voice_uses_left(user_id)

        text = (
            f"👤 *USER PROFILE*\n\n"
            f"🆔 ID: {user_id}\n"
            f"👤 First name: {user_data['first_name'] or 'Not specified'}\n"
            f"🌐 Username: @{user_data['username'] or 'Not specified'}\n\n"
            f"💫 *STATUS:*\n"
            f"• Premium: {premium_status}\n"
        )

        # Показываем откровенный режим только для премиум пользователей
        if user_data['is_premium']:
            explicit_status = "ENABLED" if user_data['explicit_mode'] else "DISABLED"
            text += f"• Explicit mode: {explicit_status}\n"

        text += f"• Voice messages left today: {voice_uses}\n\n"
        text += f"📅 Registration date: {user_data['created_at'][:10] if user_data['created_at'] else 'Unknown'}"

    bot.send_message(message.chat.id, text, parse_mode='Markdown')


@bot.message_handler(commands=['voice'])
def voice_command(message):
    """Включение/выключение голосовых сообщений"""
    user_id = message.from_user.id
    lang = user_languages[user_id]

    user_voice_enabled[user_id] = not user_voice_enabled[user_id]
    new_status = user_voice_enabled[user_id]

    if lang == 'ru':
        status = "ВКЛЮЧЕНЫ" if new_status else "ВЫКЛЮЧЕНЫ"
        text = f"🔊 Голосовые сообщения: *{status}*"
    else:
        status = "ENABLED" if new_status else "DISABLED"
        text = f"🔊 Voice messages: *{status}*"

    bot.send_message(message.chat.id, text, parse_mode='Markdown')


@bot.message_handler(commands=['clear'])
def clear_command(message):
    """Очистка истории диалога"""
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
    """Статус бота и информация о системе"""
    user_id = message.from_user.id
    lang = user_languages[user_id]

    # Получаем информацию о системе
    total_users = len(user_conversations)
    active_conversations = sum(1 for conv in user_conversations.values() if len(conv) > 0)

    if lang == 'ru':
        text = (
            f"🤖 *СТАТУС БОТА*\n\n"
            f"• 🤖 AI модель: gemini\n"
            f"• 👥 Всего пользователей: {total_users}\n"
            f"• 💬 Активных диалогов: {active_conversations}\n"
            f"*Команды:*\n"
            f"/start - начать общение\n"
            f"/profile - информация о профиле\n"
            f"/premium - премиум подписка\n"
            f"/voice - вкл/выкл голосовые\n"
            f"/clear - очистить историю\n"
            f"/status - этот статус"
        )
    else:
        text = (
            f"🤖 *BOT STATUS*\n\n"
            f"• 🤖 AI model: {ai_bot.model_name}\n"
            f"• 🎙️ Voice engine: {'Silero TTS + gTTS' if ai_bot.silero_available else 'Enhanced TTS'}\n"
            f"• 👥 Total users: {total_users}\n"
            f"• 💬 Active conversations: {active_conversations}\n"
            f"• 🗄️ Database: users.db\n\n"
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
    """Справка по командам"""
    user_id = message.from_user.id
    lang = user_languages[user_id]

    if lang == 'ru':
        text = (
            f"🤖 *ПОМОЩЬ ПО КОМАНДАМ*\n\n"
            f"*Основные команды:*\n"
            f"/start - начать общение с ботом\n"
            f"/profile - информация о вашем профиле\n"
            f"/premium - премиум подписка\n"
            f"/explicit - управление откровенным режимом\n"
            f"/voice - включить/выключить голосовые сообщения\n"
            f"/clear - очистить историю диалога\n"
            f"/status - статус бота и информация о системе\n"
            f"/help - эта справка\n\n"
            f"*Как получить голосовой ответ:*\n"
            f"Добавьте в конец сообщения: `скинь войс` или `войс`\n\n"
            f"*Лимиты:*\n"
            f"• Бесплатные пользователи: 3 войса в день\n"
            f"• Премиум пользователи: безлимитные войсы"
        )
    else:
        text = (
            f"🤖 *COMMAND HELP*\n\n"
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
            f"• Free users: 3 voice messages per day\n"
            f"• Premium users: unlimited voice messages"
        )

    bot.send_message(message.chat.id, text, parse_mode='Markdown')


# Обработчики callback-запросов
@bot.callback_query_handler(func=lambda call: call.data == 'buy_premium_stars')
def buy_premium_stars_callback(call):
    """Обработчик покупки премиум подписки через Telegram Stars"""
    user_id = call.from_user.id
    lang = user_languages[user_id]

    try:
        # Создаем инвойс для оплаты через Telegram Stars
        prices = [types.LabeledPrice(label="Premium Subscription (1 week)", amount=50)]

        # Отправляем инвойс
        bot.send_invoice(
            chat_id=call.message.chat.id,
            title="Премиум подписка на 1 неделю" if lang == 'ru' else "Premium Subscription (1 week)",
            description="Активация премиум подписки на 1 неделю. Включает безлимитные войсы, расширенную память и откровенные темы 18+" if lang == 'ru' else "Premium subscription for 1 week. Includes unlimited voice messages, enhanced memory and 18+ explicit topics",
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
            bot.answer_callback_query(call.id, "💰 Открываю окно оплаты...")
        else:
            bot.answer_callback_query(call.id, "💰 Opening payment window...")

    except Exception as e:
        logger.error(f"Ошибка при создании инвойса: {e}")
        if lang == 'ru':
            bot.answer_callback_query(call.id, "❌ Ошибка при создании платежа")
        else:
            bot.answer_callback_query(call.id, "❌ Error creating payment")


@bot.callback_query_handler(func=lambda call: call.data == 'buy_premium_crypto')
def buy_premium_crypto_callback(call):
    """Обработчик покупки премиум подписки через CryptoBot"""
    user_id = call.from_user.id
    lang = user_languages[user_id]

    # Проверяем, установлен ли токен CryptoBot
    if CRYPTO_BOT_TOKEN == "ВАШ_CRYPTOBOT_API_ТОКЕН":
        if lang == 'ru':
            bot.answer_callback_query(call.id, "❌ Оплата через CryptoBot временно недоступна")
            bot.send_message(call.message.chat.id,
                             "⚠️ Оплата через CryptoBot временно недоступна. Пожалуйста, используйте оплату через Telegram Stars.")
        else:
            bot.answer_callback_query(call.id, "❌ CryptoBot payment temporarily unavailable")
            bot.send_message(call.message.chat.id,
                             "⚠️ CryptoBot payment is temporarily unavailable. Please use Telegram Stars.")
        return

    try:
        # Создаем инвойс через CryptoBot
        invoice = create_crypto_invoice(1.0, "USDT")

        if invoice and invoice.get('pay_url'):
            pay_url = invoice['pay_url']
            invoice_id = invoice['invoice_id']

            if lang == 'ru':
                text = (
                    f"💳 *Оплата через CryptoBot*\n\n"
                    f"Сумма: *5 USDT*\n"
                    f"Срок: *1 неделя*\n\n"
                    f"Для оплаты перейдите по ссылке ниже и следуйте инструкциям.\n"
                    f"После оплаты нажмите кнопку 'Проверить оплату'.\n\n"
                    f"*Включено:*\n"
                    f"• ♾️ Безлимитные войсы\n"
                    f"• 🧠 Увеличенная память\n"
                    f"• 🔞 Откровенные темы 18+\n\n"
                    f"⚠️ *Только для пользователей 18+*"
                )
            else:
                text = (
                    f"💳 *Payment via CryptoBot*\n\n"
                    f"Amount: *5 USDT*\n"
                    f"Duration: *1 week*\n\n"
                    f"To pay, follow the link below and follow the instructions.\n"
                    f"After payment, click the 'Check Payment' button.\n\n"
                    f"*Includes:*\n"
                    f"• ♾️ Unlimited voice messages\n"
                    f"• 🧠 Enhanced memory\n"
                    f"• 🔞 18+ explicit topics\n\n"
                    f"⚠️ *For users 18+ only*"
                )

            keyboard = types.InlineKeyboardMarkup()
            keyboard.add(types.InlineKeyboardButton("🔗 Перейти к оплате", url=pay_url))
            keyboard.add(types.InlineKeyboardButton("✅ Проверить оплату", callback_data=f'check_crypto_{invoice_id}'))

            bot.send_message(call.message.chat.id, text, parse_mode='Markdown', reply_markup=keyboard)

            if lang == 'ru':
                bot.answer_callback_query(call.id, "💰 Создаем платеж...")
            else:
                bot.answer_callback_query(call.id, "💰 Creating payment...")

        else:
            logger.error(f"Не удалось создать инвойс CryptoBot: {invoice}")
            if lang == 'ru':
                bot.answer_callback_query(call.id, "❌ Ошибка при создании платежа")
                bot.send_message(call.message.chat.id,
                                 "⚠️ Не удалось создать платеж. Пожалуйста, попробуйте позже или используйте оплату через Telegram Stars.")
            else:
                bot.answer_callback_query(call.id, "❌ Error creating payment")
                bot.send_message(call.message.chat.id,
                                 "⚠️ Failed to create payment. Please try again later or use Telegram Stars.")

    except Exception as e:
        logger.error(f"Ошибка при создании CryptoBot инвойса: {e}")
        if lang == 'ru':
            bot.answer_callback_query(call.id, "❌ Ошибка при создании платежа")
        else:
            bot.answer_callback_query(call.id, "❌ Error creating payment")


@bot.callback_query_handler(func=lambda call: call.data.startswith('check_crypto_'))
def check_crypto_payment_callback(call):
    """Проверка оплаты через CryptoBot"""
    user_id = call.from_user.id
    lang = user_languages[user_id]
    invoice_id = int(call.data.split('_')[2])

    try:
        is_paid = check_crypto_payment(invoice_id)

        if is_paid:
            # Активируем премиум подписку
            user_db.activate_premium(user_id, days=7)

            if lang == 'ru':
                success_text = (
                    f"🎉 *ОПЛАТА ПОДТВЕРЖДЕНА!* 🎉\n\n"
                    f"Вы успешно активировали *ПРЕМИУМ ПОДПИСКУ* на 1 неделю!\n\n"
                    f"*Теперь вам доступно:*\n"
                    f"• ♾️ Безлимитные голосовые сообщения\n"
                    f"• 🧠 Увеличенная память диалога\n"
                    f"• 🔞 Откровенные темы для взрослых 18+\n"
                    f"• 💬 Более глубокие и интимные беседы\n\n"
                    f"Используйте /explicit чтобы управлять откровенным режимом\n\n"
                    f"Спасибо за покупку! 💫"
                )
            else:
                success_text = (
                    f"🎉 *PAYMENT CONFIRMED!* 🎉\n\n"
                    f"You have successfully activated *PREMIUM SUBSCRIPTION* for 1 week!\n\n"
                    f"*Now you have access to:*\n"
                    f"• ♾️ Unlimited voice messages\n"
                    f"• 🧠 Enhanced chat memory\n"
                    f"• 🔞 18+ explicit topics\n"
                    f"• 💬 Deeper and more intimate conversations\n\n"
                    f"Use /explicit to manage explicit mode\n\n"
                    f"Thank you for your purchase! 💫"
                )

            bot.edit_message_text(
                success_text,
                call.message.chat.id,
                call.message.message_id,
                parse_mode='Markdown'
            )

        else:
            if lang == 'ru':
                bot.answer_callback_query(call.id, "❌ Оплата не найдена. Попробуйте позже.")
            else:
                bot.answer_callback_query(call.id, "❌ Payment not found. Try again later.")

    except Exception as e:
        logger.error(f"Ошибка при проверке CryptoBot платежа: {e}")
        if lang == 'ru':
            bot.answer_callback_query(call.id, "❌ Ошибка при проверке платежа")
        else:
            bot.answer_callback_query(call.id, "❌ Error checking payment")


@bot.callback_query_handler(func=lambda call: call.data.startswith('lang_'))
def language_callback(call):
    """Обработчик выбора языка"""
    user_id = call.from_user.id
    lang = call.data.split('_')[1]
    user_languages[user_id] = lang

    if lang == 'ru':
        welcome_text = (
            f"Привет! Я Чарли - твоя виртуальная подруга 🤗\n\n"
            f"Я буду с тобой общаться, поддерживать беседу и отвечать "
            f"голосовыми сообщениями!\n\n"
            f"*Чтобы получить голосовой ответ, добавь в конец сообщения:*\n"
            f"`скинь войс` или ` войс`\n\n"
            f"*Ограничения:*\n"
            f"• Бесплатные пользователи: 3 войса в день\n"
            f"• Премиум пользователи: безлимитные войсы\n\n"
            f"💫 *Премиум подписка:* /premium - 50 Stars или CryptoBot\n\n"
            f"Расскажи мне о себе, поделись мыслями или просто поздоровайся!\n\n"
            f"*Доступные команды:*\n"
            f"/profile - информация о вашем аккаунте\n"
            f"/premium - премиум подписка\n"
            f"/explicit - управление откровенным режимом\n"
            f"/voice - вкл/выкл голосовые сообщения\n"
            f"/status - статус бота\n"
            f"/clear - очистить историю диалога\n"
            f"/help - справка по командам"
        )
    else:
        welcome_text = (
            f"Hello! I'm Charlie - your virtual girlfriend 🤗\n\n"
            f"🤖 *AI used:* {ai_bot.model_name}\n\n"
            f"🎙️ *Voice engine:* {'Silero TTS + gTTS' if ai_bot.silero_available else 'Enhanced TTS'}\n\n"
            f"I'll chat with you and sometimes respond with voice messages!\n\n"
            f"*Limitations:*\n"
            f"• Free users: 3 voice messages per day\n"
            f"• Premium users: unlimited voice messages\n\n"
            f"💫 *Premium subscription:* /premium - 50 Stars or CryptoBot\n\n"
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
    """Обработчик предварительной проверки платежа"""
    user_id = pre_checkout_query.from_user.id
    payload = pre_checkout_query.invoice_payload

    try:
        # Проверяем, что это платеж за премиум
        if payload.startswith('premium_'):
            # Подтверждаем возможность принять платеж
            bot.answer_pre_checkout_query(pre_checkout_query.id, ok=True)
        else:
            bot.answer_pre_checkout_query(pre_checkout_query.id, ok=False,
                                          error_message="Неизвестный тип платежа")
    except Exception as e:
        logger.error(f"Ошибка в pre-checkout: {e}")
        bot.answer_pre_checkout_query(pre_checkout_query.id, ok=False,
                                      error_message="Ошибка обработки платежа")


@bot.message_handler(content_types=['successful_payment'])
def successful_payment_handler(message):
    """Обработчик успешного платежа"""
    user_id = message.from_user.id
    payment_info = message.successful_payment
    lang = user_languages[user_id]

    try:
        # Активируем премиум подписку
        user_db.activate_premium(user_id, days=7)

        if lang == 'ru':
            success_text = (
                f"🎉 *ОПЛАТА ПОДТВЕРЖДЕНА!* 🎉\n\n"
                f"Вы успешно активировали *ПРЕМИУМ ПОДПИСКУ* на 1 неделю!\n\n"
                f"*Теперь вам доступно:*\n"
                f"• ♾️ Безлимитные голосовые сообщения\n"
                f"• 🧠 Увеличенная память диалога\n"
                f"• 🔞 Откровенные темы для взрослых 18+\n"
                f"• 💬 Более глубокие и интимные беседы\n\n"
                f"Используйте /explicit чтобы управлять откровенным режимом\n\n"
                f"Спасибо за покупку! 💫"
            )
        else:
            success_text = (
                f"🎉 *PAYMENT CONFIRMED!* 🎉\n\n"
                f"You have successfully activated *PREMIUM SUBSCRIPTION* for 1 week!\n\n"
                f"*Now you have access to:*\n"
                f"• ♾️ Unlimited voice messages\n"
                f"• 🧠 Enhanced chat memory\n"
                f"• 🔞 18+ explicit topics\n"
                f"• 💬 Deeper and more intimate conversations\n\n"
                f"Use /explicit to manage explicit mode\n\n"
                f"Thank you for your purchase! 💫"
            )

        bot.send_message(message.chat.id, success_text, parse_mode='Markdown')
        logger.info(f"Пользователь {user_id} активировал премиум через Stars")

    except Exception as e:
        logger.error(f"Ошибка при активации премиума после оплаты: {e}")
        if lang == 'ru':
            bot.send_message(message.chat.id, "❌ Произошла ошибка при активации премиума. Свяжитесь с поддержкой.")
        else:
            bot.send_message(message.chat.id, "❌ Error activating premium. Please contact support.")


def should_send_voice_message(user_text: str, lang: str) -> tuple:
    """Определяет, нужно ли отправлять голосовое сообщение"""
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
    """Отправляет голосовое сообщение в формате MP3"""
    try:
        with open(audio_file, 'rb') as voice_file:
            bot.send_audio(chat_id, voice_file, title="Голосовое сообщение")
        logger.info("Голосовое сообщение успешно отправлено")
        chat_voice_support[chat_id] = True
        return True
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Ошибка отправки голосового сообщения: {error_msg}")

        if "Voice_messages_forbidden" in error_msg or "voice messages are forbidden" in error_msg.lower():
            chat_voice_support[chat_id] = False
            logger.info(f"Голосовые сообщения запрещены в чате {chat_id}")
        else:
            user_voice_enabled[user_id] = False

        return False


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Обработчик всех текстовых сообщений"""
    # Пропускаем команды - они уже обработаны соответствующими обработчиками
    if message.text and message.text.startswith('/'):
        return

    user_id = message.from_user.id
    chat_id = message.chat.id
    user_text = message.text
    lang = user_languages[user_id]

    logger.info(f"Получено сообщение от {user_id}: {user_text}")

    # Убедимся, что пользователь есть в базе
    if not user_db.get_user(user_id):
        user_db.create_user(user_id, message.from_user.username,
                            message.from_user.first_name, message.from_user.last_name)

    # Проверяем, запрошено ли голосовое сообщение
    send_voice, cleaned_text = should_send_voice_message(user_text, lang)

    # Проверяем условия для отправки голосового
    can_send_voice = (
            user_voice_enabled[user_id] and
            chat_voice_support[chat_id] and
            send_voice and
            user_db.can_use_voice(user_id)
    )

    # Если текст пустой после удаления триггера
    if not cleaned_text.strip():
        cleaned_text = "Привет" if lang == 'ru' else "Hello"

    # Показываем индикатор набора
    bot.send_chat_action(chat_id, 'typing')

    # Получаем ответ от AI с учетом того, запрошено ли голосовое
    bot_response = ai_bot.get_ai_response(user_id, cleaned_text, lang, voice_requested=send_voice)

    # Отправляем голосовое если нужно
    if can_send_voice:
        audio_file = ai_bot.text_to_speech(bot_response, user_id, lang)
        if audio_file:
            try:
                voice_success = send_voice_message(chat_id, audio_file, user_id)

                if voice_success:
                    # Увеличиваем счетчик использований войсов
                    user_db.increment_voice_use(user_id)

                    # Показываем сколько войсов осталось
                    user_data = user_db.get_user(user_id)
                    if not user_data['is_premium']:
                        uses_left = 3 - user_data['voice_uses_today']
                        if uses_left > 0:
                            if lang == 'ru':
                                reminder = f"ℹ️ Осталось войсов сегодня: {uses_left}/3\n💫 Безлимитные войсы с /premium"
                            else:
                                reminder = f"ℹ️ Voice messages left today: {uses_left}/3\n💫 Unlimited voice with /premium"
                            bot.send_message(chat_id, reminder)

                if not voice_success:
                    if chat_voice_support[chat_id]:
                        if lang == 'ru':
                            bot.send_message(chat_id, "⚠️ Не удалось отправить голосовое сообщение.")
                        else:
                            bot.send_message(chat_id, "⚠️ Couldn't send voice message.")
                    else:
                        if lang == 'ru':
                            bot.send_message(chat_id, "ℹ️ В этом чате голосовые сообщения запрещены.")
                        else:
                            bot.send_message(chat_id, "ℹ️ Voice messages are forbidden in this chat.")

            except Exception as e:
                logger.error(f"Ошибка при обработке голосового сообщения: {e}")
                # В случае ошибки отправляем текстовый ответ
                bot.send_message(chat_id, bot_response)
            finally:
                if os.path.exists(audio_file):
                    try:
                        os.remove(audio_file)
                    except Exception as e:
                        logger.error(f"Ошибка при удалении файла: {e}")
        else:
            # Если не удалось создать голосовое, отправляем текстовый ответ
            bot.send_message(chat_id, bot_response)
    else:
        # Если голосовое не запрошено или недоступно, отправляем текстовый ответ
        bot.send_message(chat_id, bot_response)

        # Если запрошено голосовое, но превышен лимит
        if send_voice and not user_db.can_use_voice(user_id):
            user_data = user_db.get_user(user_id)
            if not user_data['is_premium']:
                if lang == 'ru':
                    bot.send_message(
                        chat_id,
                        f"❌ Лимит войсов исчерпан!3/3 войсов сегодня.\n\n"
                        f"💫 Премиум пользователи имеют безлимитные войсы!\n"
                        f"Используйте /premium для активации за 50 звезд"
                    )
                else:
                    bot.send_message(
                        chat_id,
                        f"❌ Voice message limit reached! You've used 3/3 voice messages today.\n\n"
                        f"💫 *Premium users* get unlimited voice messages!\n"
                        f"Use /premium to activate for 50 Telegram Stars"
                    )

        if send_voice and not chat_voice_support[chat_id]:
            if lang == 'ru':
                bot.send_message(chat_id, "ℹ️ В этом чате голосовые сообщения запрещены.")
            else:
                bot.send_message(chat_id, "ℹ️ Voice messages are forbidden in this chat.")


if __name__ == '__main__':
    # Проверяем конфигурацию перед запуском
    validate_config()

    print("=" * 50)
    print("🤖 Бот Шарлотта запускается...")
    print(f"🤖 Используемый AI: {ai_bot.model_name}")
    print(f"🎙️ Голосовой движок: {'Silero TTS + gTTS' if ai_bot.silero_available else 'Улучшенный TTS'}")
    print(f"💾 База данных: users.db")
    print(f"💫 Система оплаты: Telegram Stars + CryptoBot")
    print(f"🔞 Премиум режим: управление откровенными темами")
    print("=" * 50)

    try:
        bot.infinity_polling()
    except Exception as e:
        logger.error(f"Критическая ошибка бота: {e}")
        print(f"Критическая ошибка: {e}")