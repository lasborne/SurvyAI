import asyncio
import httpx
from survyai_cloud.db import init_db
from survyai_cloud.main import app

async def main():
    await init_db()
    tr = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=tr, base_url="http://t") as c:
        print("health", (await c.get("/health")).status_code)
        r = await c.post(
            "/v1/auth/register",
            json={"email": "paystack2@test.com", "password": "password12"},
        )
        print("register", r.status_code)

asyncio.run(main())
