"""init

Revision ID: 66d82a122d77
Revises: 
Create Date: 2024-11-15 15:22:47.499534

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '66d82a122d77'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('image_result',
    sa.Column('created_at', mysql.TIMESTAMP(), server_default=sa.text('current_timestamp'), nullable=False),
    sa.Column('updated_at', mysql.TIMESTAMP(), server_default=sa.text('current_timestamp on update current_timestamp'), nullable=False),
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('file_path', sa.String(length=255), nullable=True),
    sa.Column('file_name', sa.String(length=255), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('id')
    )
    op.create_table('object_predicted',
    sa.Column('created_at', mysql.TIMESTAMP(), server_default=sa.text('current_timestamp'), nullable=False),
    sa.Column('updated_at', mysql.TIMESTAMP(), server_default=sa.text('current_timestamp on update current_timestamp'), nullable=False),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('accuracy', sa.Float(), nullable=True),
    sa.Column('label', sa.String(length=255), nullable=True),
    sa.Column('xmin', sa.Float(), nullable=True),
    sa.Column('ymin', sa.Float(), nullable=True),
    sa.Column('xmax', sa.Float(), nullable=True),
    sa.Column('ymax', sa.Float(), nullable=True),
    sa.Column('image_result_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['image_result_id'], ['image_result.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('object_predicted')
    op.drop_table('image_result')
    # ### end Alembic commands ###