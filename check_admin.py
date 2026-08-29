import sys
sys.path.insert(0, 'c:\\Users\\marie\\Desktop\\Projets\\PFE Issat\\backend')

from app.database import SessionLocal
from app.models import User

db = SessionLocal()
admin = db.query(User).filter(User.email == 'admin@example.com').first()
if admin:
    print(f"Admin user found: {admin.email}, Role: {admin.role}, Active: {admin.is_active}")
else:
    print("Admin user not found")
db.close()
