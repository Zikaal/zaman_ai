from models import db, User, Transaction
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
from decimal import Decimal
import random
import time
from sqlalchemy.orm import sessionmaker
import traceback

# Подключение к той же БД
engine = create_engine('postgresql://postgres.decwxdskxcrehiauwjyq:fLXxkf42l6NtY@aws-1-us-east-2.pooler.supabase.com:6543/postgres')
Session = sessionmaker(bind=engine)

def simulate_expenses():
    while True:
        session = Session()
        try:
            # Явная проверка и загрузка пользователей
            users = session.query(User).options(db.joinedload(User.goals)).all()
            for user in users:
                categories = ['еда', 'такси', 'развлечения']
                amount = Decimal(str(random.uniform(1000, 5000)))
                category = random.choice(categories)
                description = f'Симулированный расход на {category}'
                trans = Transaction(
                    user_id=user.id,
                    type='expense',
                    amount=amount,
                    category=category,
                    description=description
                )
                session.add(trans)
                user.balance -= amount
                print(f"Added expense for user {user.id}: {amount} in {category}")
            session.commit()
        except SQLAlchemyError as e:
            print(f"Background error: {e}")
            traceback.print_exc()
            session.rollback()
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            session.rollback()
        finally:
            session.close()
        time.sleep(2)

if __name__ == '__main__':
    simulate_expenses()