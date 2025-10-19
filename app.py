from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from models import db, User, Goal, Transaction, Product, ChatHistory # Предполагается, что models.py существует
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import requests
import json
import numpy as np
import plotly.express as px
import plotly.io as pio
import traceback
import time
import re
from decimal import Decimal
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
import os

# Добавлены импорты для Gemini SDK
from google import genai
from google.genai import types

load_dotenv()

# Инициализация Flask и расширений
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'super_secret_key'
db.init_app(app)
login_manager = LoginManager(app)
bcrypt = Bcrypt(app)

# Обновленные настройки API для Gemini
# *** ЗАМЕНИТЕ ЭТОТ ПЛЕЙСХОЛДЕР НА ВАШ РЕАЛЬНЫЙ GEMINI API КЛЮЧ ***
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_BASE = 'https://generativelanguage.googleapis.com/v1'
GEMINI_CHAT_MODEL = 'gemini-2.5-flash'
GEMINI_EMBEDDING_MODEL = 'text-embedding-004'

# Общие заголовки с API ключом
GEMINI_HEADERS = {
    'Content-Type': 'application/json'
}

# Инициализация клиента Gemini SDK
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Failed to initialize Gemini Client: {e}")
    gemini_client = None

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Setup requests session with retry strategy
def create_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session
requests_session = create_session()

# ========== HELPERS ==========
def cosine_similarity(a, b):
    try:
        if not a or not b:
            return 0.0
        # Преобразование в numpy array и нормализация (добавление 1e-10 для стабильности)
        a_norm = np.array(a, dtype=float) / (np.linalg.norm(a) + 1e-10)
        b_norm = np.array(b, dtype=float) / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a_norm, b_norm))
    except Exception as e:
        print(f"Cosine similarity error: {e}")
        return 0.0

def get_embedding(text, retries=3):
    """Получить embedding для Gemini API"""
    if not text or not isinstance(text, str):
        return [0] * 768 
    
    text = text[:3072] 
    
    for attempt in range(retries):
        try:
            payload = {
                "model": GEMINI_EMBEDDING_MODEL, 
                "content": {"parts": [{"text": text}]},
                "task_type": "RETRIEVAL_DOCUMENT"
            }
            url = f"{GEMINI_API_BASE}/models/{GEMINI_EMBEDDING_MODEL}:embedContent?key={GEMINI_API_KEY}"
            
            response = requests_session.post(
                url,
                headers=GEMINI_HEADERS,
                json=payload,
                timeout=15,
                verify=True
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'embedding' in data and 'values' in data['embedding']:
                    return data['embedding']['values']
            
            print(f"Embedding attempt {attempt+1}: Status {response.status_code}, Response: {response.text}")
            if attempt < retries - 1:
                time.sleep(1 + attempt)
        except requests.exceptions.RequestException as e:
            print(f"Embedding request error attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f"Embedding error attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(1)
                
    return [0] * 768 

def get_product_recommendations(user_message, timeout=10):
    """Получить рекомендации продуктов на основе векторов"""
    try:
        if not user_message:
            return "Пожалуйста, опишите ваши финансовые потребности."
        
        user_emb = get_embedding(user_message)
        # Проверяем, что эмбеддинг не состоит из нулей
        if np.allclose(user_emb, [0] * len(user_emb), atol=1e-5):
            print("ERROR: User embedding is nearly zero. Gemini embedding API might be failing or key is wrong.")
            return "Ошибка: не удалось получить эмбеддинг пользователя."
            
        products = Product.query.limit(50).all() 
        
        if not products:
            print("ERROR: Product database is empty.")
            return "В базе нет доступных продуктов. Попробуйте позже."
        
        # Фильтрация по ключевым словам для начального отбора
        user_lower = user_message.lower()
        type_matches = []
        if any(word in user_lower for word in ['депозит', 'сбережения', 'накопления', 'инвестиции']):
            type_matches = [p for p in products if 'депозит' in p.type.lower() or 'сбережения' in p.type.lower() or 'вакала' in p.type.lower()]
        elif any(word in user_lower for word in ['ипотека', 'квартира', 'жилье', 'финансирование', 'мурабаха']):
            type_matches = [p for p in products if 'ипотека' in p.type.lower() or 'финансирование' in p.type.lower() or 'мурабаха' in p.type.lower()]
        elif any(word in user_lower for word in ['бизнес', 'карта', 'расчетный', 'платежный']):
            type_matches = [p for p in products if 'кредит' in p.type.lower() or 'мурабаха' in p.type.lower() or 'платежный' in p.type.lower()]
        else:
            type_matches = products

        scores = []
        
        for prod in type_matches[:15]: 
            try:
                prod_desc = f"{prod.name} {prod.type} {prod.description or ''}".lower()
                prod_emb = get_embedding(prod_desc)
                
                if np.allclose(prod_emb, [0] * len(prod_emb), atol=1e-5):
                    continue 
                    
                score = cosine_similarity(user_emb, prod_emb)
                
                # ИСПРАВЛЕНО: Снижение порогового значения для релевантности
                if score > 0.10: # Порог снижен до 0.10
                    yield_text = f"доходность {prod.expected_yield}%" if prod.expected_yield else ""
                    reason = f"{prod.name} ({prod.type}) — {yield_text} (score: {score:.2f})"
                    scores.append((prod, score, reason))
                    
            except Exception as e:
                print(f"Product processing error: {e}")
                continue
                
        scores.sort(key=lambda x: x[1], reverse=True)
        top_products = scores[:2] 
        
        if top_products:
            rec_text = "; ".join([reason for _, _, reason in top_products])
            return f"Рекомендации: {rec_text}"
            
        return "Нет подходящих продуктов для вашего запроса."
        
    except Exception as e:
        print(f"Recommendation error: {e}")
        traceback.print_exc()
        return "Ошибка при анализе продуктов."

def llm_chat(messages, empathetic=True, user_id=None, retries=2):
    """LLM chat с обработкой ошибок и таймаутами для Gemini API"""
    if not messages:
        return "Нет сообщений для обработки."
    
    gemini_messages = []
    
    # УСИЛЕННЫЙ СИСТЕМНЫЙ ПРОМПТ
    prompt = (
        "Ты empathetic AI-ассистент банка Zaman, говори на русском. "
        "Используй исламские термины (Мурабаха вместо кредит). "
        "**ОБЯЗАТЕЛЬНО** проанализируй данные в тегах <КОНТЕКСТ_РЕКОМЕНДАЦИЙ>. "
        "Предложи 1-2 продукта, которые ты нашел в <КОНТЕКСТ_РЕКОМЕНДАЦИЙ>, если они присутствуют. "
        "Если тегов <КОНТЕКСТ_РЕКОМЕНДАЦИЙ> нет или они пустые, закончи советы без упоминания продуктов. "
        "Отвечай вежливо и кратко."
    )
    
    user_message_with_system = ""
    if empathetic and messages and messages[0].get('role') == 'user':
        user_message_with_system = f"{prompt}\n\n{messages[0]['content']}"
    
    rec_text = ""
    # Добавляем рекомендации в контекст последнего сообщения
    if messages and messages[-1]['role'] == 'user':
        user_input_for_rec = messages[-1]['content']
        try:
            rec_text = get_product_recommendations(user_input_for_rec, timeout=5)
            
            # ИСПРАВЛЕННЫЙ ФОРМАТ ДОБАВЛЕНИЯ КОНТЕКСТА С ТЕГАМИ
            if rec_text and "Рекомендации" in rec_text:
                clean_rec_data = rec_text.replace("Рекомендации: ", "").strip()
                context_to_add = f"\n\n<КОНТЕКСТ_РЕКОМЕНДАЦИЙ>{clean_rec_data}</КОНТЕКСТ_РЕКОМЕНДАЦИЙ>"

                if user_message_with_system:
                    user_message_with_system += context_to_add
                elif messages[-1]['role'] == 'user':
                    messages[-1]['content'] += context_to_add

        except Exception as e:
            print(f"Recommendations error: {e}")

    # Формирование массива 'contents' для Gemini
    for i, msg in enumerate(messages):
        role = 'user' if msg.get('role') == 'user' else 'model'
        content = msg.get('content')
        
        if i == 0 and role == 'user' and user_message_with_system:
            content = user_message_with_system
        
        gemini_messages.append({
            "role": role, 
            "parts": [{"text": content}]
        })
    
    if not gemini_messages:
         return "Нет сообщений для обработки."

    for attempt in range(retries):
        try:
            payload = {
                "contents": gemini_messages,
                "generationConfig": {
                    "temperature": 0.5,
                    "maxOutputTokens": 5024
                }
            }
            url = f"{GEMINI_API_BASE}/models/{GEMINI_CHAT_MODEL}:generateContent?key={GEMINI_API_KEY}"
            
            response = requests_session.post(
                url,
                headers=GEMINI_HEADERS,
                json=payload,
                timeout=20,
                verify=True
            )
            
            if response.status_code == 200:
                data = response.json()
                try:
                    ai_response = data['candidates'][0]['content']['parts'][0]['text']
                except (KeyError, IndexError) as e:
                    reason = data['candidates'][0].get('finishReason', 'UNKNOWN')
                    if reason == 'MAX_TOKENS':
                        ai_response = "Извините, ответ был обрезан, поскольку достигнут лимит слов. Пожалуйста, повторите запрос или сделайте его короче."
                    else:
                        print(f"Response parsing error with reason: {reason}")
                        ai_response = "Извините, не удалось обработать ответ модели."
                        
                if user_id and messages[-1]['role'] == 'user':
                    try:
                        # Чистим сообщение пользователя от контекста перед сохранением в БД
                        clean_user_message = re.sub(r'<КОНТЕКСТ_РЕКОМЕНДАЦИЙ>.*?</КОНТЕКСТ_РЕКОМЕНДАЦИЙ>', '', messages[-1]['content'], flags=re.DOTALL).strip()
                        clean_user_message = re.sub(r'Contextual Recommendations.*', '', clean_user_message, flags=re.DOTALL).strip()
                        
                        history = ChatHistory(
                            user_id=user_id,
                            user_message=clean_user_message,
                            ai_response=ai_response
                        )
                        db.session.add(history)
                        db.session.commit()
                    except Exception as db_error:
                        print(f"Database save error: {db_error}")
                        db.session.rollback()
                        
                return ai_response
            
            print(f"LLM attempt {attempt+1}: Status {response.status_code}, Response: {response.text}")
            if attempt < retries - 1:
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"LLM request error attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(3)
        except Exception as e:
            print(f"LLM error attempt {attempt+1}: {e}")
            traceback.print_exc()
            if attempt < retries - 1:
                time.sleep(2)
                
    return "Извините, не удалось получить ответ. Попробуйте позже."

# ========== ROUTES ==========

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    if db.session.query(User).filter_by(username=username).first():
        return jsonify({'error': 'Username taken'}), 400
    
    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    user = User(username=username, password_hash=hashed, balance=0.0)
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'Registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    user = db.session.query(User).filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password_hash, password):
        login_user(user)
        return jsonify({'message': 'Logged in successfully'}), 200
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout', methods=['GET'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200


@app.route('/chat', methods=['POST'])
def chat():
    """Обновленный роут для чата"""
    try:
        data = request.json
        messages = data.get('messages', [])
        user_id = data.get('user_id')
        if not messages or not isinstance(messages, list):
            return jsonify({'error': 'Valid messages array required'}), 400
        
        if GEMINI_API_KEY == 'YOUR_GEMINI_API_KEY':
            return jsonify({'error': 'API Key is not set in the code'}), 500

        response = llm_chat(messages, user_id=user_id)
        return jsonify({'response': response}), 200
    except Exception as e:
        print(f"Chat error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Chat processing error'}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Роут для транскрибирования и чата с использованием Gemini API."""
    if not gemini_client:
        return jsonify({'error': 'Gemini Client not initialized. Check API Key.'}), 500
        
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No audio file found in request'}), 400
            
        audio_file = request.files['file']
        if not audio_file.filename:
            return jsonify({'error': 'No file selected'}), 400
            
        # 1. Чтение файла в байты и определение MIME-типа
        audio_bytes = audio_file.read()
        mime_type = audio_file.mimetype or 'audio/mpeg' # Используем 'audio/mpeg' как запасной вариант

        # Ограничение размера файла для inline-загрузки (около 20MB)
        if len(audio_bytes) > (20 * 1024 * 1024):
            return jsonify({'error': 'Audio file is too large for inline processing (max 20MB)'}), 400

        # 2. Формирование мультимодального запроса для Gemini
        prompt_parts = [
            "Transcribe the audio provided below. Only output the text of the transcription.",
            # Передаем аудио как часть (Part)
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type,
            ),
        ]

        # 3. Вызов Gemini API для транскрипции
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_parts,
            config=types.GenerateContentConfig(
                temperature=0.0 # Низкая температура для точной транскрипции
            )
        )

        text = response.text.strip()
        
        if not text:
             return jsonify({'error': 'Gemini returned an empty transcription.'}), 500

        # 4. Передача транскрибированного текста в LLM для чата
        user_id = request.args.get('user_id')
        chat_resp = llm_chat([{"role": "user", "content": text}], user_id=user_id)
        
        return jsonify({'transcribed': text, 'response': chat_resp}), 200

    except Exception as e:
        print(f"Gemini Transcription error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Server error during Gemini transcription: {str(e)}'}), 500

# ... (Остальные роуты set_goal, add_expense, add_income, user, products, /, errorhandler)

@app.route('/set_goal', methods=['POST'])
@login_required
def set_goal():
    try:
        data = request.json
        goal_type = data.get('goal_type', '').strip()
        cost = float(data.get('cost', 0))
        timeline = int(data.get('timeline', 12))
        if not goal_type or cost <= 0:
            return jsonify({'error': 'Valid goal_type and cost required'}), 400
        goal = Goal(
            user_id=current_user.id,
            goal_type=goal_type,
            cost=cost,
            timeline=timeline
        )
        db.session.add(goal)
        db.session.commit()
        goal_text = f"Цель: {goal_type} на сумму {cost} тг за {timeline} мес."
        response = llm_chat([{"role": "user", "content": goal_text}], user_id=current_user.id)
        return jsonify({'message': 'Goal set', 'response': response}), 200
    except ValueError:
        return jsonify({'error': 'Invalid data types'}), 400
    except Exception as e:
        db.session.rollback()
        print(f"Set goal error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/add_expense', methods=['POST'])
@login_required
def add_expense():
    try:
        data = request.json
        amount = Decimal(str(data.get('amount', 0)))
        trans = Transaction(
            user_id=current_user.id,
            type='expense',
            amount=amount,
            category=data.get('category'),
            description=data.get('description')
        )
        db.session.add(trans)
        current_user.balance -= amount
        db.session.commit()
        expense_text = f"Добавлен расход: {data.get('category', 'неизвестно')} на {data.get('amount', 0)} тг."
        messages = [{"role": "user", "content": expense_text}]
        response = llm_chat(messages, user_id=current_user.id)
        return jsonify({'message': 'Expense added', 'response': response}), 200
    except Exception as e:
        print(f"Add expense error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/add_income', methods=['POST'])
@login_required
def add_income():
    try:
        data = request.json
        amount = Decimal(str(data.get('amount', 0)))
        trans = Transaction(
            user_id=current_user.id,
            type='income',
            amount=amount,
            category=data.get('category'),
            description=data.get('description')
        )
        db.session.add(trans)
        current_user.balance += amount
        db.session.commit()
        income_text = f"Добавлен доход: {data.get('category', 'неизвестно')} на {data.get('amount', 0)} тг."
        messages = [{"role": "user", "content": income_text}]
        response = llm_chat(messages, user_id=current_user.id)
        return jsonify({'message': 'Income added', 'response': response}), 200
    except Exception as e:
        print(f"Add income error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_expenses', methods=['POST'])
@login_required
def analyze_expenses():
    try:
        data = request.json
        expenses_text = json.dumps(data.get('expenses', []), ensure_ascii=False)
        prompt = f"Анализируй расходы: {expenses_text}. Предложи смену привычек, альтернативы стрессу (не траты), будь empathetic."
        messages = [{"role": "user", "content": prompt}]
        response = llm_chat(messages, user_id=current_user.id)
        return jsonify({'analysis': response, 'message': 'Analysis saved'}), 200
    except Exception as e:
        print(f"Analyze expenses error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/user', methods=['GET'])
@login_required
def get_user():
    return jsonify({
        'username': current_user.username,
        'balance': float(current_user.balance),
        'goals': [{'type': g.goal_type, 'cost': float(g.cost), 'timeline': g.timeline} for g in current_user.goals],
        'transactions': [{'type': t.type, 'amount': float(t.amount), 'category': t.category, 'description': t.description, 'timestamp': t.timestamp.isoformat()} for t in current_user.transactions]
    }), 200

@app.route('/products', methods=['GET'])
def get_products():
    try:
        products = Product.query.all()
        return jsonify([p.to_dict() for p in products]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommendations', methods=['GET'])
@login_required
def get_recommendations():
    try:
        history = ChatHistory.query.filter_by(user_id=str(current_user.id)).order_by(ChatHistory.timestamp.desc()).limit(5).all()
        recs = []
        for h in history:
            if any(word in h.ai_response.lower() for word in ['рекомендую', 'товар', 'продукт']):
                recs.append({'user_message': h.user_message, 'recommendation': h.ai_response})
        return jsonify(recs), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize')
def visualize():
    try:
        user_id = request.args.get('user_id', current_user.id if current_user.is_authenticated else 'default')
        history = ChatHistory.query.filter_by(user_id=user_id).all()
        if not history:
            return jsonify({'error': 'No history'}), 404
        
        df = pd.DataFrame([h.to_dict() for h in history])
        if not df.empty:
            fig = px.bar(df, x='timestamp', y='id', title='Chat History Length')
            return jsonify({'chart': pio.to_json(fig)}), 200
        return jsonify({'chart': None}), 200
    except Exception as e:
        print(f"Visualize error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/visualize_goal_progress', methods=['GET'])
@login_required
def visualize_goal_progress():
    try:
        goals = Goal.query.filter_by(user_id=current_user.id).all()
        if not goals:
            return jsonify({'error': 'No goals found for the user'}), 404

        # Prepare data for visualization
        data = []
        for goal in goals:
            accumulated = float(current_user.balance)  # Using balance as accumulated amount
            remaining = float(goal.cost) - accumulated if float(goal.cost) > accumulated else 0
            data.append({
                'goal_type': goal.goal_type,
                'accumulated': accumulated if accumulated > 0 else 0,
                'remaining': remaining,
                'total': float(goal.cost)
            })

        # Create a DataFrame for Plotly
        df = pd.DataFrame(data)
        if df.empty:
            return jsonify({'error': 'No valid goal data to visualize'}), 404

        # Create a stacked bar chart
        fig = px.bar(
            df,
            x='goal_type',
            y=['accumulated', 'remaining'],
            title='Progress Towards Goals',
            labels={'value': 'Amount (KZT)', 'goal_type': 'Goal Type', 'variable': 'Status'},
            barmode='stack'
        )
        fig.update_layout(
            yaxis_title="Amount (KZT)",
            xaxis_title="Goal Type",
            legend_title="Progress Status"
        )

        return jsonify({'chart': pio.to_json(fig)}), 200
    except Exception as e:
        print(f"Visualize goal progress error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/compare')
def compare():
    try:
        user_id = request.args.get('user_id', current_user.id if current_user.is_authenticated else 'default')
        history = ChatHistory.query.filter_by(user_id=user_id).all()
        if not history:
            return jsonify({'error': 'No history'}), 404
        
        num_questions = len([h for h in history if len(h.user_message) > 10])
        num_recommendations = len([h for h in history if any(word in h.ai_response.lower() for word in ['рекомендую', 'товар', 'продукт'])])
        comparison = f"Вопросов юзера: {num_questions}. Рекомендаций ИИ: {num_recommendations}."
        return jsonify({'comparison': comparison}), 200
    except Exception as e:
        print(f"Compare error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.errorhandler(500)
def handle_error(error):
    traceback.print_exc()
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
        except SQLAlchemyError as e:
            print(f"Database creation error: {e}")
            
    app.run(debug=False, host='0.0.0.0', port=10000, threaded=True)