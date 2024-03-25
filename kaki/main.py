from kaki.datafeed.handler.okx_crypto_download import AsyncCryptoDataUpdater
import asyncio

if __name__ == '__main__':
    updater = AsyncCryptoDataUpdater()
    asyncio.run(updater.main())
