from argparse import ArgumentParser
from pymilvus import MilvusClient
import random

BASE = 1000000

def generate_data(num_entries, dim, offset=0):
    return [
        {
            "id": i + offset,
            "vector": [random.uniform(-1, 1) for _ in range(dim)],
            "color": f"color_{random.randint(1000, 9999)}",
        }
        for i in range(num_entries)
    ]

def ensure_collection(client, collection_name, dim):
    if not client.has_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            dimension=dim,
        )

def insert_data(client, collection_name, num_entries, dim):
    offset = 0
    remain = num_entries
    while remain > 0:
        batch_size = min(remain, BASE)
        data_batch = generate_data(batch_size, dim, offset)
        result = client.insert(
            collection_name=collection_name,
            data=data_batch
        )
        print(result)
        offset += batch_size
        remain -= batch_size

def run_milvus(opts):
    client = MilvusClient("http://localhost:19530")
    ensure_collection(client, opts.collection, opts.dim)
    insert_data(client, opts.collection, opts.num, opts.dim)

def main():
    parser = ArgumentParser(description="Milvus data insertion script.")
    parser.add_argument("-n", "--num", type=int, default=1000000, help="Number of entries to insert")
    parser.add_argument("-d", "--dim", type=int, default=5, help="Dimension of vector field")
    parser.add_argument("-c", "--collection", default="test", help="Name of the collection to use")
    parser.add_argument("-m", "--metric", default="IP", help="Metric of similarity search")
    parser.add_argument("-i", "--index", default="HNSW", help="Vector index")
    args = parser.parse_args()
    print(args)
    run_milvus(args)

if __name__ == "__main__":
    main()

