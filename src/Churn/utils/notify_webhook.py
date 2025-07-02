from typing import Dict

import httpx

from .logging import logger


HEADERS_JSON: Dict[str, str] = {"x-api-key": "X-API_KEY", "Content-Type": "application/json"}


async def post_to_webhook(webhook_url: str, WEBHOOK_PAYLOAD: Dict, headers: Dict[str, str] = HEADERS_JSON) -> bool:
    """
    Post a notification to the webhook if the input is valid.

    Raises:
        Exception: If posting to the webhook fails.

    Returns:
        True if the webhook is notified successfully.
    """

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.post(webhook_url, json=WEBHOOK_PAYLOAD, headers=headers)
            if response.status_code in (200, 201):
                logger.info(f"Background task completed and webhook sent. Payload: {WEBHOOK_PAYLOAD}")
                return True
            else:
                raise Exception(f"Error notifying webhook: status code {response.status_code}")
        except Exception as e:
            logger.error(f"Exception posting to webhook: {str(e)}")
            raise
