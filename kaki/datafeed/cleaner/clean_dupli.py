from pymongo import MongoClient

def clean_duplicates(collection):
    # Ensure an index on the 'timestamp' field for faster lookups
    collection.create_index([('timestamp', 1)])

    # Stage 1: Identify all duplicate timestamps
    pipeline = [
    {
        '$group': {
            '_id': '$timestamp',  # Group by the 'timestamp' field
            'uniqueIds': {'$addToSet': '$_id'},  # Collect all IDs for each timestamp
            'count': {'$sum': 1}  # Count occurrences
        }
    },
    {
        '$match': {
            'count': {'$gt': 1}  # Select groups with more than one occurrence
        }
    }
    ]

    duplicates = list(collection.aggregate(pipeline))

    # Stage 2: Use bulk operations to remove duplicates, keeping only the first document for each timestamp
    bulk = collection.initialize_ordered_bulk_op()
    for duplicate in duplicates:
    # Skip the first ID to keep one document for each timestamp
        for dup_id in duplicate['uniqueIds'][1:]:
            bulk.find({'_id': dup_id}).remove_one()

        # Execute bulk operation
        if duplicates:  # Only execute if there are duplicates
            result = bulk.execute()
            print(f"Deleted {result['nRemoved']} duplicate documents.")
        else:
            print("No duplicates found.")
