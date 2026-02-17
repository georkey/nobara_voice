import asyncio
import subprocess

import dbus_next

async def healthcheck():
    print("Starting...")
    await asyncio.create_subprocess_exec("sleep", "3")
    print("After sleep")

if __name__ == "__main__":
    asyncio.run(healthcheck())