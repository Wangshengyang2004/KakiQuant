"""
Functions to check database and updates augmently
"""
def get_crypto_info(instType="SPOT"):
    import okx.PublicData as PublicData

    flag = "0"  # 实盘:0 , 模拟盘：1

    publicDataAPI = PublicData.PublicAPI(flag=flag)

    # 获取交易产品基础信息
    result = publicDataAPI.get_instruments(
        instType
    )
    return result["data"]

def get_ideal_date_range(symbol):
    pass

def get_collection_date_range(collection):
    """
    Returns the earliest (start) and latest (end) timestamps in the specified MongoDB collection.
    
    Parameters:
    - collection: A pymongo collection object.
    
    Returns:
    - A tuple containing the start and end timestamps.
    """
    # Aggregate to find the minimum and maximum timestamps
    pipeline = [
        {
            "$group": {
                "_id": None,
                "start_date": {"$min": "$timestamp"},
                "end_date": {"$max": "$timestamp"}
            }
        }
    ]
    
    result = list(collection.aggregate(pipeline))
    
    if result:
        start_date = result[0]['start_date']
        end_date = result[0]['end_date']
        return start_date, end_date
    else:
        return None, None
    

if __name__ == "__main__":
    info = get_crypto_info(instType="SWAP")