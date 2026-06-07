import logging

import uvicorn

from receipt_extractor.api.app import app
from receipt_extractor.logging_config import setup_logging

if __name__ == "__main__":
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Start Backend Service")

    uvicorn.run(app, host="0.0.0.0", port=8080)
