import asyncio
import logging

logger = logging.getLogger()


async def healthcheck():
    print("Starting...")
    await asyncio.create_subprocess_exec("sleep", "3")
    print("After sleep")


async def set_volume(vol: int):
    cmd = await asyncio.create_subprocess_exec(
        "pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{vol}%"
    )
    stdout, stderr = await cmd.communicate()
    exit_code = cmd.returncode
    logger.info(f"{stdout}")
    if exit_code != 0:
        logger.error(f"Error with code {exit_code} in set_volume")
        logger.error(f"{stderr}")


if __name__ == "__main__":
    asyncio.run(healthcheck())
