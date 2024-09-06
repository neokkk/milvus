from pymilvus import MilvusClient

COLLECTION_NAME = "test"
NUM_DELETE = 1000000

client = MilvusClient("http://localhost:19530")

ids = list(range(0, NUM_DELETE))

res = client.delete(
    collection_name=COLLECTION_NAME,
    ids=ids,
)
print(res)
