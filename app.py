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

import os
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL") 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'super_secret_key'
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

API_BASE = os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")
HEADERS = {'Authorization': f'Bearer {API_KEY}'}

# ========== HELPERS ==========
def cosine_similarity(a, b):
    try:
        a_norm = np.array(a) / (np.linalg.norm(a) + 1e-10)
        b_norm = np.array(b) / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a_norm, b_norm))
    except Exception:
        return 0.0

def get_embedding(text, retries=3):
    for attempt in range(retries):
        try:
            payload = {"model": "text-embedding-3-small", "input": text}
            response = requests.post(f"{API_BASE}/embeddings", headers=HEADERS, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()['data'][0]['embedding']
            print(f"Embedding attempt {attempt+1}: Status {response.status_code}")
            time.sleep(2)
        except Exception as e:
            print(f"Embedding attempt {attempt+1} error: {e}")
            time.sleep(2)
    return [0] * 1536

def get_product_recommendations(user_message):
    try:
        user_emb = get_embedding(user_message)
        products = Product.query.all()
        if not products:
            return "Нет продуктов в базе Supabase. Заполните таблицу products для рекомендаций."

        user_lower = user_message.lower()
        type_matches = []
        if any(word in user_lower for word in ['депозит', 'сбережения', 'накопления', 'инвестиции', 'овернайт', 'выгодный']):
            type_matches = [p for p in products if 'депозитный' in p.type.lower()]
        elif any(word in user_lower for word in ['ипотека', 'квартира', 'жилье', 'покупка', 'финансирование', 'рассрочка']):
            type_matches = [p for p in products if 'финансирование' in p.type.lower() or 'ипотека' in p.name.lower()]
        elif any(word in user_lower for word in ['бизнес', 'карта', 'овер', 'кредит', 'платежный', 'тариф', 'расчёт']):
            type_matches = [p for p in products if 'кредит' in p.type.lower() or 'платежный' in p.type.lower() or 'расчётно' in p.type.lower() or 'бизнес' in p.name.lower()]
        else:
            type_matches = products

        scores = []
        for prod in type_matches:
            prod_desc = f"{prod.name} {prod.type} {prod.description}".lower()
            prod_emb = get_embedding(prod_desc)
            score = cosine_similarity(user_emb, prod_emb)
            if score > 0.2:
                yield_text = f"доходность {prod.expected_yield}" if prod.expected_yield else ""
                markup_text = f"наценка от {prod.min_markup} тг" if hasattr(prod, 'min_markup') else ""
                reason = f"{prod.name} ({prod.type}) — {yield_text}, {markup_text} (score: {score:.2f}). Шариат-compliant."
                scores.append((prod, score, reason))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_products = scores[:2] if scores else []

        if top_products:
            rec_text = "; ".join([reason for _, _, reason in top_products])
            return f"Рекомендации: {rec_text}"
        return "Нет подходящих продуктов в Supabase для вашего запроса."
    except Exception as e:
        print(f"Recommendation error: {e}")
        return "Ошибка при анализе продуктов. Попробуйте позже."

def llm_chat(messages, empathetic=True, user_id=None, retries=3):
    for attempt in range(retries):
        try:
            prompt = "Ты empathetic AI-ассистент банка Zaman, говори на русском, используй исламские термины (Мурабаха вместо кредит, без риба/процентов). Будь человечным, мотивируй, предлагай альтернативы стрессу (медитация, не траты). Анализируй цели/привычки персонально. Сначала дай общие советы по запросу пользователя (например, планирование бюджета, терпение, медитация). В конце предложи 1-2 подходящих продукта из базы Supabase, объясняя, почему они подходят (доходность, сроки, суммы, шариат-compliant). Используй ТОЛЬКО продукты из рекомендаций, не придумывай новые. Если подходящих нет, закончи советы без упоминания продуктов."
            if empathetic:
                messages.insert(0, {"role": "system", "content": prompt})

            if messages and messages[-1]['role'] == 'user':
                user_message = messages[-1]['content']
                rec_text = get_product_recommendations(user_message)
                messages[0]["content"] += f"\n{rec_text}. Начни с общих советов, затем, если есть рекомендации, предложи 1-2 продукта в конце с обоснованием."

            payload = {"model": "gpt-4o-mini", "messages": messages, "temperature": 0.5}
            response = requests.post(f"{API_BASE}/chat/completions", headers=HEADERS, json=payload, timeout=30)
            if response.status_code == 200:
                ai_response = response.json()['choices'][0]['message']['content']
                if user_id:
                    history = ChatHistory(
                        user_id=user_id,
                        user_message=messages[-1]['content'] if messages[-1]['role'] == 'user' else "Системное",
                        ai_response=ai_response
                    )
                    db.session.add(history)
                    db.session.commit()
                return ai_response
            print(f"LLM attempt {attempt+1}: Status {response.status_code}")
            time.sleep(2)
        except Exception as e:
            print(f"LLM attempt {attempt+1} error: {e}")
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
        user_id = data.get('user_id', 'default_user')
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
        
        audio_file = request.files['file']
        files = {'file': (audio_file.filename, audio_file, 'audio/mpeg')}
        data_payload = {'model': 'whisper-1'}
        
        response = requests.post(f"{API_BASE}/audio/transcriptions", headers=HEADERS, files=files, data=data_payload, timeout=30)
        if response.status_code == 200:
            text = response.json()['text']
            user_id = request.args.get('user_id', 'transcribe_user')
            chat_resp = llm_chat([{"role": "user", "content": text}], user_id=user_id)
            return jsonify({'transcribed': text, 'response': chat_resp}), 200
        else:
            return jsonify({'error': response.text}), 500
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
        amount = Decimal(str(data.get('amount', 0)))  # Convert float to Decimal
        trans = Transaction(
            user_id=current_user.id,
            type='expense',
            amount=amount,
            category=data.get('category'),
            description=data.get('description')
        )
        db.session.add(trans)
        current_user.balance -= amount  # Now both are Decimal
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
        amount = Decimal(str(data.get('amount', 0)))  # Convert to Decimal
        trans = Transaction(
            user_id=current_user.id,
            type='income',
            amount=amount,  # Use Decimal for consistency
            category=data.get('category'),
            description=data.get('description')
        )
        db.session.add(trans)
        current_user.balance += amount  # Now both are Decimal
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