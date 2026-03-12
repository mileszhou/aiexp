import logging

# 1. Get a named logger (best practice: use __name__ so logs know which module sent them)
logger = logging.getLogger(__name__)

# 2. Configure once (usually in main script or entry point)
#    This example logs to console + file, INFO level and above
logging.basicConfig(
    level=logging.INFO,                     # Show INFO and higher (DEBUG skipped unless changed)
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),            # → console
        logging.FileHandler("app.log")      # → file (appends by default)
    ]
)

# 3. Now use it anywhere (even in other modules)
def do_something(user_id: int):
    logger.debug("Very detailed info (usually hidden)")     # level 10
    logger.info("Processing user %d", user_id)              # level 20
    logger.warning("Deprecated API used by user %d", user_id)  # level 30
    try:
        1 / 0
    except ZeroDivisionError:
        logger.error("Division by zero occurred", exc_info=True)  # level 40 + stack trace
        logger.critical("System may be unstable now!")            # level 50

if __name__ == "__main__":
    logger.info("Application starting")
    do_something(123)
    logger.info("Application finished")