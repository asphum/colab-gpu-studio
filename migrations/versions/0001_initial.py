"""Independent initial database; no changes to Atelier tables."""
from alembic import op
from app.db import Base
from app.gpu import repository
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    Base.metadata.create_all(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError('เพื่อปกป้องงาน สัญญาเช่า และผลงาน ระบบไม่รองรับการดาวน์เกรดแบบทำลายข้อมูลโดยอัตโนมัติ ให้ใช้การกู้คืนจากข้อมูลสำรองไปยังฐานข้อมูลทดสอบแยกต่างหาก')
