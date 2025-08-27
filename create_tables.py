# create_tables.py
import asyncio
from admin_app import engine, Base

async def run():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Таблицы созданы!")

asyncio.run(run())
