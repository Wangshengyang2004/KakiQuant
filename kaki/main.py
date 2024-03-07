import asyncio
# Maintain DB
from kaki.datafeed.update.async_crypto import AsyncCryptoDataUpdater
updater = AsyncCryptoDataUpdater()
asyncio.run(updater.main())
# Setup Website

