import asyncio
import logging

from command_service.dbus_commands import set_volume

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


async def main():
    await set_volume(48)


if __name__ == "__main__":
    logger.info("Starting...")
    asyncio.run(main())
