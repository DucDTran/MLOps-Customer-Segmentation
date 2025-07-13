import redis
import json

class RedisFeatureStore:
    
    def __init__(self, redis_host: str="localhost", redis_port: int=6379, redis_db: int=0):
        
        self.client = redis.StrictRedis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
        )

    def store_features(self, entity_id: str, features: dict):

        key = f"entity:{entity_id}:features"
        self.client.set(key, json.dumps(features))

    def get_features(self, entity_id: str) -> dict:

        key = f"entity:{entity_id}:features"
        features = self.client.get(key)

        if features is None:
            return None
        return json.loads(features)
    
    def store_batch_features(self, batch_data: dict):
        for entity_id, features in batch_data.items():
            self.store_features(entity_id, features)
    
    def get_batch_features(self, entity_ids: list[str]):
        batch_features = {}
        for entity_id in entity_ids:
            features = self.get_features(entity_id)
            if features is not None:
                batch_features[entity_id] = features
        return batch_features
    
    def delete_features(self, entity_id: str):
        key = f"entity:{entity_id}:features"
        self.client.delete(key)

    def get_all_entity_ids(self):
        keys = self.client.keys('entity:*:features')
        entity_ids = [key.split(':')[1] for key in keys]
        return entity_ids