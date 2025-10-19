from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from models import db, User, Goal, Transaction, Product, ChatHistory
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

from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.getenv('SECRET_KEY', 'super_secret_key')
db.init_app(app)

CORS(app, resources={
    r"/*": {
        "origins": ["https://zaman-bank.vercel.app", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response, 200

login_manager = LoginManager(app)
bcrypt = Bcrypt(app)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Инициализация Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY not set in environment variables!")

try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("✓ Gemini client initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize Gemini Client: {e}")
    gemini_client = None

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
        a_norm = np.array(a, dtype=float) / (np.linalg.norm(a) + 1e-10)
        b_norm = np.array(b, dtype=float) / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a_norm, b_norm))
    except Exception as e:
        print(f"Cosine similarity error: {e}")
        return 0.0

def get_embedding(text, retries=3):
    """Получить embedding через SDK (более надежно)"""
    if not text or not isinstance(text, str):
        return [0] * 768 
    
    if not gemini_client:
        print("ERROR: Gemini client not initialized")
        return [0] * 768
    
    text = text[:3072].strip()
    
    for attempt in range(retries):
        try:
            response = gemini_client.models.embed_content(
                model='text-embedding-004',
                content=text,
                config=types.EmbedContentConfig(
                    task_type='RETRIEVAL_DOCUMENT'
                )
            )
            
            if hasattr(response, 'embedding') and response.embedding:
                return list(response.embedding)
            
            print(f"Embedding attempt {attempt+1}: Empty response")
            time.sleep(1 + attempt)
            
        except Exception as e:
            print(f"Embedding attempt {attempt+1} error: {type(e).__name__}: {e}")
            time.sleep(1 + attempt)
                
    print("ERROR: Could not get embedding after retries")
    return [0] * 768

def get_product_recommendations(user_message):
    try:
        if not user_message:
            return "Пожалуйста, опишите ваши финансовые потребности."
        
        user_emb = get_embedding(user_message)
        if np.allclose(user_emb, [0] * len(user_emb), atol=1e-5):
            print("WARNING: User embedding is nearly zero")
            return "Рекомендуем обновить ваш профиль для лучших рекомендаций."
            
        products = Product.query.limit(50).all() 
        
        if not products:
            return "В базе нет доступных продуктов. Попробуйте позже."
        
        user_lower = user_message.lower()
        type_matches = []
        if any(word in user_lower for word in ['депозит', 'сбережения', 'накопления', 'инвестиции']):
            type_matches = [p for p in products if 'депозит' in p.type.lower() or 'сбережения' in p.type.lower()]
        elif any(word in user_lower for word in ['ипотека', 'квартира', 'жилье', 'финансирование', 'мурабаха']):
            type_matches = [p for p in products if 'ипотека' in p.type.lower() or 'финансирование' in p.type.lower()]
        elif any(word in user_lower for word in ['бизнес', 'карта', 'расчетный', 'платежный']):
            type_matches = [p for p in products if 'кредит' in p.type.lower() or 'мурабаха' in p.type.lower()]
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
                
                if score > 0.10:
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
    """LLM chat с Gemini SDK"""
    if not messages:
        return "Нет сообщений для обработки."
    
    if not gemini_client:
        return "Ошибка: AI ассистент временно недоступен."
    
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
    if messages and messages[-1]['role'] == 'user':
        user_input_for_rec = messages[-1]['content']
        try:
            rec_text = get_product_recommendations(user_input_for_rec)
            
            if rec_text and "Рекомендации" in rec_text:
                clean_rec_data = rec_text.replace("Рекомендации: ", "").strip()
                context_to_add = f"\n\n<КОНТЕКСТ_РЕКОМЕНДАЦИЙ>{clean_rec_data}</КОНТЕКСТ_РЕКОМЕНДАЦИЙ>"

                if user_message_with_system:
                    user_message_with_system += context_to_add
                elif messages[-1]['role'] == 'user':
                    messages[-1]['content'] += context_to_add

        except Exception as e:
            print(f"Recommendations error: {e}")

    gemini_messages = []
    for i, msg in enumerate(messages):
        role = 'user' if msg.get('role') == 'user' else 'model'
        content = msg.get('content')
        
        if i == 0 and role == 'user' and user_message_with_system:
            content = user_message_with_system
        
        gemini_messages.append(types.Content(
            role=role,
            parts=[types.Part.from_text(content)]
        ))
    
    if not gemini_messages:
        return "Нет сообщений для обработки."

    for attempt in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=gemini_messages,
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=6024
                )
            )
            
            ai_response = response.text.strip() if response.text else "Не удалось получить ответ."
            
            if user_id and messages[-1]['role'] == 'user':
                try:
                    clean_user_message = re.sub(r'<КОНТЕКСТ_РЕКОМЕНДАЦИЙ>.*?</КОНТЕКСТ_РЕКОМЕНДАЦИЙ>', '', messages[-1]['content'], flags=re.DOTALL).strip()
                    
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
            
        except Exception as e:
            print(f"LLM error attempt {attempt+1}: {type(e).__name__}: {e}")
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
    try:
        data = request.json
        messages = data.get('messages', [])
        user_id = data.get('user_id')
        if not messages:
            return jsonify({'error': 'messages required'}), 400
        response = llm_chat(messages, user_id=user_id)
        return jsonify({'response': response}), 200
    except Exception as e:
        print(f"Chat error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No audio file'}), 400
        
        if not gemini_client:
            return jsonify({'error': 'AI service unavailable'}), 503
        
        audio_file = request.files['file']
        audio_bytes = audio_file.read()
        mime_type = audio_file.mimetype or 'audio/mpeg'
        
        if len(audio_bytes) > (20 * 1024 * 1024):
            return jsonify({'error': 'Audio file is too large (max 20MB)'}), 400

        prompt_parts = [
            "Transcribe the audio provided below. Only output the text of the transcription.",
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type,
            ),
        ]

        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_parts,
            config=types.GenerateContentConfig(temperature=0.0)
        )

        text = response.text.strip() if response.text else ""
        
        if not text:
            return jsonify({'error': 'Transcription failed'}), 500

        user_id = request.args.get('user_id')
        chat_resp = llm_chat([{"role": "user", "content": text}], user_id=user_id)
        
        return jsonify({'transcribed': text, 'response': chat_resp}), 200

    except Exception as e:
        print(f"Transcribe error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/set_goal', methods=['POST'])
@login_required
def set_goal():
    try:
        data = request.json
        goal = Goal(
            user_id=current_user.id,
            goal_type=data.get('goal_type'),
            cost=float(data.get('cost', 0)),
            timeline=int(data.get('timeline', 12))
        )
        db.session.add(goal)
        db.session.commit()
        goal_text = f"Цель: {data.get('goal_type', 'неизвестно')} на сумму {data.get('cost', 0)} тг за {data.get('timeline', 12)} мес."
        messages = [{"role": "user", "content": goal_text}]
        response = llm_chat(messages, user_id=current_user.id)
        return jsonify({'message': 'Goal set', 'response': response}), 200
    except Exception as e:
        print(f"Set goal error: {e}")
        traceback.print_exc()
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

        data = []
        for goal in goals:
            accumulated = float(current_user.balance)
            remaining = float(goal.cost) - accumulated if float(goal.cost) > accumulated else 0
            data.append({
                'goal_type': goal.goal_type,
                'accumulated': accumulated if accumulated > 0 else 0,
                'remaining': remaining,
                'total': float(goal.cost)
            })

        df = pd.DataFrame(data)
        if df.empty:
            return jsonify({'error': 'No valid goal data to visualize'}), 404

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
        db.create_all()
    app.run(debug=True)