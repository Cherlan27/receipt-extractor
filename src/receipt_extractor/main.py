import logging

import uvicorn
from api.app import app
from logging_config import setup_logging

if __name__ == "__main__":
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Start Backend Service")

    uvicorn.run(app, host="0.0.0.0", port=8080)
