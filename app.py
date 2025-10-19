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
from functools import lru_cache

from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JSON_SORT_KEYS'] = False
app.secret_key = 'super_secret_key'
db.init_app(app)

CORS(app, resources={
    r"/*": {
        "origins": ["https://zaman-bank.vercel.app", "http://localhost:3000", "https://openai-hub.neuraldeep.tech/v1-A"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Global OPTIONS handler
@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Connection'] = 'keep-alive'
        return response, 200

@app.after_request
def after_request(response):
    response.headers['Connection'] = 'keep-alive'
    response.headers['Keep-Alive'] = 'timeout=60, max=100'
    return response

login_manager = LoginManager(app)
bcrypt = Bcrypt(app)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

API_BASE = os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")
HEADERS = {'Authorization': f'Bearer {API_KEY}'}

# ========== CONSTANTS ==========
REQUEST_TIMEOUT = 90
EMBEDDING_TIMEOUT = 60
LLM_TIMEOUT = 90
MAX_RETRIES = 3
RETRY_DELAY = 2

# ========== HELPERS ==========
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    try:
        a_norm = np.array(a) / (np.linalg.norm(a) + 1e-10)
        b_norm = np.array(b) / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a_norm, b_norm))
    except Exception as e:
        print(f"Cosine similarity error: {e}")
        return 0.0

@lru_cache(maxsize=128)
def get_embedding(text, retries=MAX_RETRIES):
    """Get embedding from API with caching and retry logic"""
    if not API_BASE or not API_KEY:
        return [0] * 1536
    
    for attempt in range(retries):
        try:
            payload = {"model": "text-embedding-3-small", "input": text}
            response = requests.post(
                f"{API_BASE}/embeddings",
                headers=HEADERS,
                json=payload,
                timeout=EMBEDDING_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()['data'][0]['embedding']
            
            if response.status_code == 429:  # Rate limit
                wait_time = int(response.headers.get('Retry-After', RETRY_DELAY * (attempt + 1)))
                print(f"Embedding rate limited, waiting {wait_time}s (attempt {attempt+1}/{retries})")
                time.sleep(wait_time)
                continue
            
            print(f"Embedding attempt {attempt+1}/{retries}: Status {response.status_code}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
                
        except requests.exceptions.Timeout:
            print(f"Embedding timeout on attempt {attempt+1}/{retries}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"Embedding error on attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
    
    print(f"Failed to get embedding for text after {retries} attempts")
    return [0] * 1536

def get_product_recommendations(user_message, limit=2):
    """Get product recommendations based on user message"""
    try:
        products = Product.query.all()
        if not products:
            return "Нет продуктов в базе Supabase. Заполните таблицу products для рекомендаций."

        # Quick keyword matching first (no embedding call)
        user_lower = user_message.lower()
        type_matches = []
        
        if any(word in user_lower for word in ['депозит', 'сбережения', 'накопления', 'инвестиции', 'овернайт', 'выгодный']):
            type_matches = [p for p in products if 'депозитный' in p.type.lower()]
        elif any(word in user_lower for word in ['ипотека', 'квартира', 'жилье', 'покупка', 'финансирование', 'рассрочка']):
            type_matches = [p for p in products if 'финансирование' in p.type.lower() or 'ипотека' in p.name.lower()]
        elif any(word in user_lower for word in ['бизнес', 'карта', 'овер', 'кредит', 'платежный', 'тариф', 'расчёт']):
            type_matches = [p for p in products if 'кредит' in p.type.lower() or 'платежный' in p.type.lower() or 'расчётно' in p.type.lower() or 'бизнес' in p.name.lower()]
        else:
            type_matches = products[:5]  # Limit to avoid too many embeddings

        if not type_matches:
            type_matches = products[:3]

        # Get embedding once for user message
        user_emb = get_embedding(user_message)
        
        scores = []
        # Limit embeddings to avoid timeout
        for prod in type_matches[:5]:
            try:
                prod_desc = f"{prod.name} {prod.type} {prod.description}".lower()
                prod_emb = get_embedding(prod_desc)
                score = cosine_similarity(user_emb, prod_emb)
                
                if score > 0.2:
                    yield_text = f"доходность {prod.expected_yield}%" if prod.expected_yield else ""
                    markup_text = f"наценка от {prod.min_markup} тг" if hasattr(prod, 'min_markup') and prod.min_markup else ""
                    details = f"{yield_text}, {markup_text}".strip(", ")
                    reason = f"{prod.name} ({prod.type}) — {details} (score: {score:.2f}). Шариат-compliant."
                    scores.append((prod, score, reason))
            except Exception as e:
                print(f"Error scoring product {prod.name}: {e}")
                continue

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        top_products = scores[:limit]

        if top_products:
            rec_text = "; ".join([reason for _, _, reason in top_products])
            return f"Рекомендации: {rec_text}"
        
        return "Нет подходящих продуктов в Supabase для вашего запроса."
        
    except Exception as e:
        print(f"Recommendation error: {e}")
        traceback.print_exc()
        return "Ошибка при анализе продуктов. Попробуйте позже."

def llm_chat(messages, empathetic=True, user_id=None, retries=MAX_RETRIES):
    """Call LLM with retry logic and better error handling"""
    if not API_BASE or not API_KEY:
        return "Server misconfiguration: API_BASE or API_KEY not set on server."
    
    if not messages:
        return "No messages provided"
    
    for attempt in range(retries):
        try:
            # Build system prompt
            system_prompt = "Ты empathetic AI-ассистент банка Zaman, говори на русском, используй исламские термины (Мурабаха вместо кредит, без риба/процентов). Будь человечным, мотивируй, предлагай альтернативы стрессу (медитация, не траты). Анализируй цели/привычки персонально. Сначала дай общие советы по запросу пользователя (например, планирование бюджета, терпение, медитация). В конце предложи 1-2 подходящих продукта из базы Supabase, объясняя, почему они подходят (доходность, сроки, суммы, шариат-compliant). Используй ТОЛЬКО продукты из рекомендаций, не придумывай новые. Если подходящих нет, закончи советы без упоминания продуктов."
            
            # Prepare messages with system prompt
            messages_to_send = []
            if empathetic:
                messages_to_send.append({"role": "system", "content": system_prompt})
            
            messages_to_send.extend(messages)

            # Get product recommendations if user message exists
            if messages_to_send and messages_to_send[-1]['role'] == 'user':
                user_message = messages_to_send[-1]['content']
                rec_text = get_product_recommendations(user_message)
                
                # Update system prompt with recommendations
                if messages_to_send[0]['role'] == 'system':
                    messages_to_send[0]["content"] += f"\n{rec_text}. Начни с общих советов, затем, если есть рекомендации, предложи 1-2 продукта в конце с обоснованием."

            # Make API call with timeout
            payload = {
                "model": "gpt-4o-mini",
                "messages": messages_to_send,
                "temperature": 0.5,
                "max_tokens": 1024
            }
            
            response = requests.post(
                f"{API_BASE}/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=LLM_TIMEOUT
            )
            
            if response.status_code == 200:
                ai_response = response.json()['choices'][0]['message']['content']
                
                # Save to chat history
                if user_id:
                    try:
                        history = ChatHistory(
                            user_id=user_id,
                            user_message=messages_to_send[-1]['content'] if messages_to_send[-1]['role'] == 'user' else "Системное",
                            ai_response=ai_response
                        )
                        db.session.add(history)
                        db.session.commit()
                    except Exception as e:
                        print(f"Error saving chat history: {e}")
                        db.session.rollback()
                
                return ai_response
            
            elif response.status_code == 429:  # Rate limit
                wait_time = int(response.headers.get('Retry-After', RETRY_DELAY * (attempt + 1)))
                print(f"LLM rate limited, waiting {wait_time}s (attempt {attempt+1}/{retries})")
                time.sleep(wait_time)
                continue
            
            else:
                print(f"LLM attempt {attempt+1}/{retries}: Status {response.status_code}")
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY)
        
        except requests.exceptions.Timeout:
            print(f"LLM timeout on attempt {attempt+1}/{retries}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
        
        except Exception as e:
            print(f"LLM error on attempt {attempt+1}/{retries}: {e}")
            traceback.print_exc()
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
    
    return "Извините, не удалось получить ответ. Попробуйте позже."

# ========== ROUTES ==========
@app.route('/register', methods=['POST'])
def register():
    try:
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
        
        return jsonify({'message': 'Registered successfully', 'user_id': user.id}), 201
    except Exception as e:
        db.session.rollback()
        print(f"Register error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        user = db.session.query(User).filter_by(username=username).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            return jsonify({'message': 'Logged in successfully', 'user_id': user.id}), 200
        
        return jsonify({'error': 'Invalid credentials'}), 401
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': str(e)}), 500

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
        
        if not isinstance(messages, list):
            return jsonify({'error': 'messages must be a list'}), 400
        
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
        
        audio_file = request.files['file']
        if not audio_file.filename:
            return jsonify({'error': 'Empty filename'}), 400
        
        files = {'file': (audio_file.filename, audio_file, 'audio/mpeg')}
        data_payload = {'model': 'whisper-1'}
        
        response = requests.post(
            f"{API_BASE}/audio/transcriptions",
            headers=HEADERS,
            files=files,
            data=data_payload,
            timeout=EMBEDDING_TIMEOUT
        )
        
        if response.status_code == 200:
            text = response.json().get('text', '')
            user_id = request.args.get('user_id')
            
            chat_resp = llm_chat([{"role": "user", "content": text}], user_id=user_id)
            return jsonify({'transcribed': text, 'response': chat_resp}), 200
        else:
            return jsonify({'error': f'Transcription failed: {response.text}'}), 500
    
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Transcription request timeout'}), 504
    except Exception as e:
        print(f"Transcribe error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/set_goal', methods=['POST'])
@login_required
def set_goal():
    try:
        data = request.json
        goal_type = data.get('goal_type')
        cost = data.get('cost', 0)
        timeline = data.get('timeline', 12)
        
        if not goal_type:
            return jsonify({'error': 'goal_type required'}), 400
        
        goal = Goal(
            user_id=current_user.id,
            goal_type=goal_type,
            cost=float(cost),
            timeline=int(timeline)
        )
        db.session.add(goal)
        db.session.commit()
        
        goal_text = f"Цель: {goal_type} на сумму {cost} тг за {timeline} мес."
        messages = [{"role": "user", "content": goal_text}]
        response = llm_chat(messages, user_id=current_user.id)
        
        return jsonify({'message': 'Goal set', 'response': response, 'goal_id': goal.id}), 200
    
    except Exception as e:
        db.session.rollback()
        print(f"Set goal error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/add_expense', methods=['POST'])
@login_required
def add_expense():
    try:
        data = request.json
        amount = Decimal(str(data.get('amount', 0)))
        category = data.get('category', 'Other')
        description = data.get('description', '')
        
        if amount <= 0:
            return jsonify({'error': 'Amount must be positive'}), 400
        
        trans = Transaction(
            user_id=current_user.id,
            type='expense',
            amount=amount,
            category=category,
            description=description
        )
        db.session.add(trans)
        current_user.balance -= amount
        db.session.commit()
        
        expense_text = f"Добавлен расход: {category} на {amount} тг."
        messages = [{"role": "user", "content": expense_text}]
        response = llm_chat(messages, user_id=current_user.id)
        
        return jsonify({'message': 'Expense added', 'response': response}), 200
    
    except Exception as e:
        db.session.rollback()
        print(f"Add expense error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/add_income', methods=['POST'])
@login_required
def add_income():
    try:
        data = request.json
        amount = Decimal(str(data.get('amount', 0)))
        category = data.get('category', 'Other')
        description = data.get('description', '')
        
        if amount <= 0:
            return jsonify({'error': 'Amount must be positive'}), 400
        
        trans = Transaction(
            user_id=current_user.id,
            type='income',
            amount=amount,
            category=category,
            description=description
        )
        db.session.add(trans)
        current_user.balance += amount
        db.session.commit()
        
        income_text = f"Добавлен доход: {category} на {amount} тг."
        messages = [{"role": "user", "content": income_text}]
        response = llm_chat(messages, user_id=current_user.id)
        
        return jsonify({'message': 'Income added', 'response': response}), 200
    
    except Exception as e:
        db.session.rollback()
        print(f"Add income error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_expenses', methods=['POST'])
@login_required
def analyze_expenses():
    try:
        data = request.json
        expenses = data.get('expenses', [])
        
        if not expenses:
            return jsonify({'error': 'expenses required'}), 400
        
        expenses_text = json.dumps(expenses, ensure_ascii=False)
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
    try:
        return jsonify({
            'id': current_user.id,
            'username': current_user.username,
            'balance': float(current_user.balance),
            'goals': [
                {
                    'id': g.id,
                    'type': g.goal_type,
                    'cost': float(g.cost),
                    'timeline': g.timeline
                }
                for g in current_user.goals
            ],
            'transactions': [
                {
                    'id': t.id,
                    'type': t.type,
                    'amount': float(t.amount),
                    'category': t.category,
                    'description': t.description,
                    'timestamp': t.timestamp.isoformat()
                }
                for t in current_user.transactions
            ]
        }), 200
    except Exception as e:
        print(f"Get user error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/products', methods=['GET'])
def get_products():
    try:
        products = Product.query.all()
        return jsonify([p.to_dict() for p in products]), 200
    except Exception as e:
        print(f"Get products error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/recommendations', methods=['GET'])
@login_required
def get_recommendations():
    try:
        history = ChatHistory.query.filter_by(user_id=str(current_user.id)).order_by(ChatHistory.timestamp.desc()).limit(5).all()
        recs = []
        
        for h in history:
            if any(word in h.ai_response.lower() for word in ['рекомендую', 'товар', 'продукт']):
                recs.append({
                    'user_message': h.user_message,
                    'recommendation': h.ai_response,
                    'timestamp': h.timestamp.isoformat()
                })
        
        return jsonify(recs), 200
    except Exception as e:
        print(f"Get recommendations error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/visualize')
def visualize():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            if current_user.is_authenticated:
                user_id = current_user.id
            else:
                return jsonify({'error': 'user_id required or login required'}), 400
        
        history = ChatHistory.query.filter_by(user_id=str(user_id)).all()
        if not history:
            return jsonify({'error': 'No history'}), 404
        
        df = pd.DataFrame([h.to_dict() for h in history])
        if not df.empty:
            fig = px.bar(df, x='timestamp', y='id', title='Chat History Length')
            return jsonify({'chart': pio.to_json(fig)}), 200
        
        return jsonify({'chart': None}), 200
    
    except Exception as e:
        print(f"Visualize error: {e}")
        traceback.print_exc()
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
                'accumulated': max(0, accumulated),
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
        user_id = request.args.get('user_id')
        if not user_id:
            if current_user.is_authenticated:
                user_id = current_user.id
            else:
                return jsonify({'error': 'user_id required or login required'}), 400
        
        history = ChatHistory.query.filter_by(user_id=str(user_id)).all()
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

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'details': str(error)}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'details': str(error)}), 404

@app.errorhandler(500)
def handle_error(error):
    traceback.print_exc()
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)