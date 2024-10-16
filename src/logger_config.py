import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger('RhetoresGraeci')

if __name__ == "__main__":
    logger.info("Logger for Rhetores Graeci initialized.")