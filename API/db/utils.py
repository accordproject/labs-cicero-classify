def convert_mongo_id(data):
    data["id"] = str(data["_id"])
    del data["_id"]
    return data