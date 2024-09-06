import os
import sys
#home_dir = os.environ.get("HOME")
#sys.path.append("/usr/lib/python3/dist-packages")
#sys.path.append("/usr/local/lib/python3.10/dist-packages")
#sys.path.append(f"{home_dir}/.local/lib/python3-dist-packages")
#print(sys.path)

import argparse
from collections import namedtuple
import h5py
from multiprocessing.pool import ThreadPool
import numpy as np
import psutil
from pymilvus import DataType, connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from prometheus_api_client import PrometheusConnect
import requests
import signal
import subprocess
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

from datasets import DATASETS, get_dataset

class BaseANN(object):
    def done(self) -> None:
        pass

    def get_memory_usage(self) -> Optional[float]:
        return psutil.Process().memory_info().rss / 1024

    def fit(self, X: np.array) -> None:
        pass

    def query(self, q: np.array, n: int) -> np.array:
       return []  # array of candidate indices

    def batch_query(self, X: np.array, n: int) -> None:
        pool = ThreadPool()
        self.res = pool.map(lambda q: self.query(q, n), X)

    def get_batch_results(self) -> np.array:
        return self.res

    def get_additional(self) -> Dict[str, Any]:
        return {}

    def __str__(self) -> str:
        return self.name

def metric_mapping(_metric: str):
    print(f"_metric: {_metric}")
    _metric_type = {"angular": "COSINE", "euclidean": "L2"}.get(_metric, None)
    if _metric_type is None:
        raise Exception(f"[Milvus] Not support metric type: {_metric}!!!")
    return _metric_type

def get_pid(process_name):
    try:
        pid = subprocess.check_output(['pgrep', '-f', process_name]).decode().strip()
        return pid
    except subprocess.CalledProcessError:
        return None

process = None
promConn = None
promSession = None

def connect_prometheus():
    promSession = requests.Session()
    try:
        promConn = PrometheusConnect(session=promSession, disable_ssl=True)
        print("[Prometheus] connect session to prometheus server")

    except Exception as e:
        print("[Prometheus] Fail to connect to prometheus server:", e)
        promSession.close()

def disconnect_prometheus():
    print("[Prometheus] disconnect session")
    if promSession:
        promSession.close()

class Milvus(BaseANN):
    def __init__(self, metric, dim, index_param, local, skip):
        print("Milvus init local: ", local)
        self._metric = metric
        self._dim = dim
        self.local = local
        self.skip = skip
        self._metric_type = metric_mapping(self._metric)
        connect_prometheus()
        if skip:
            pid = get_pid("milvus")
            process = namedtuple("process", ["pid"])
            self.p = process(pid)
            print("is skipped. pid:", process.pid)
        else:
            self.start_milvus()
        self.connects = connections
        max_trys = 10
        for try_num in range(max_trys):
            try:
                self.connects.connect("default", host="localhost", port="19530")
                break
            except Exception as e:
                if try_num == max_trys - 1:
                    raise Exception(f"[Milvus] connect to milvus failed: {e}!!!")
                print(f"[Milvus] try to connect to milvus again...")
                time.sleep(1)
        print(f"[Milvus] Milvus version: {utility.get_server_version()}")
        self.collection_name = "test_milvus"
        if utility.has_collection(self.collection_name):
            print(f"[Milvus] collection {self.collection_name} already exists, drop it...")
            utility.drop_collection(self.collection_name)

    def __str__(self):
        return f"name: {self.name}, dim: {self._dim}, metric: {self._metric}"

    def start_milvus(self):
        try:
            if self.local:
                home_path = "/home/nk"
                milvus_path = f"{home_path}/milvus"
                milvus_binpath = f"{milvus_path}/bin/milvus"
                env = os.environ.copy()
                ld_path = f"{milvus_path}/internal/core/output/lib:lib"
                print("ld_path: ", ld_path)
                env["LD_LIBRARY_PATH"] = ld_path
                with open(os.devnull, "w") as devnull:
                    process = subprocess.Popen([milvus_binpath, "run", "standalone"], env=env, stdout=devnull, stderr=devnull, text=True)
                    # process = subprocess.Popen([milvus_binpath, "run", "standalone"], env=env, stdout=sys.stdout, stderr=sys.stderr, text=True)
                    self.p = process
                    print(f"PID: {process.pid}")
            else:
                os.system("docker compose down")
                os.system("docker compose up -d")
                print("[Milvus] docker compose up successfully!!!")
        except Exception as e:
            print(f"[Milvus] start up failed: {e}!!!")

    def stop_milvus(self):
        try:
            if self.local and self.p:
                print(f"Kill process {self.p.pid}")
                self.p.kill()
            elif self.skip:
                print(f"Kill process {self.p.pid}")
                os.kill(self.p.pid, signal.SIGKILL)
            else:
                os.system("docker compose down")
            print("[Milvus] docker compose down successfully!!!")
        except Exception as e:
            print(f"[Milvus] docker compose down failed: {e}!!!")

    def create_collection(self):
        print("[Milvus] Create collection start")
        filed_id = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True
        )
        filed_vec = FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=self._dim
        )
        schema = CollectionSchema(
            fields=[filed_id, filed_vec],
            description="Test milvus search",
        )
        self.collection = Collection(
            self.collection_name,
            schema,
            consistence_level="STRONG"
        )
        print(f"[Milvus] Create collection {self.collection.describe()} successfully!!!")

    def insert(self, X):
        # insert data
        print(f"[Milvus] Insert {len(X)} data into collection {self.collection_name}...")
        batch_size = 1000
        for i in range(0, len(X), batch_size):
            batch_data = X[i: min(i + batch_size, len(X))]
            entities = [
                [i for i in range(i, min(i + batch_size, len(X)))],
                batch_data.tolist()
            ]
            self.collection.insert(entities)
        self.collection.flush()
        print(f"[Milvus] {self.collection.num_entities} data has been inserted into collection {self.collection_name}!!!")

    def get_index_param(self):
        raise NotImplementedError()

    def create_index(self):
        # create index
        print(f"[Milvus] Create index for collection {self.collection_name}...")
        self.collection.create_index(
            field_name = "vector",
            index_params = self.get_index_param(),
            index_name = "vector_index"
        )
        utility.wait_for_index_building_complete(
            collection_name = self.collection_name,
            index_name = "vector_index"
        )
        index = self.collection.index(index_name = "vector_index")
        index_progress =  utility.index_building_progress(
            collection_name = self.collection_name,
            index_name = "vector_index"
        )
        print(f"[Milvus] Create index {index.to_dict()} {index_progress} for collection {self.collection_name} successfully!!!")

    def load_collection(self):
        # load collection
        print(f"[Milvus] Load collection {self.collection_name}...")
        self.collection.load()
        utility.wait_for_loading_complete(self.collection_name)
        print(f"[Milvus] Load collection {self.collection_name} successfully!!!")

    def fit(self, X):
        start_build_time = time.time()
        self.create_collection()
        self.insert(X)
        end_build_time = time.time()
        print(f"build time: {end_build_time - start_build_time}")
        self.create_index()
        end_index_time = time.time()
        print(f"build index time: {end_index_time - end_build_time}")
        self.load_collection()

    def query(self, v, n):
        results = self.collection.search(
            data = [v],
            anns_field = "vector",
            param = self.search_params,
            limit = n,
            output_fields=["id"]
        )
        ids = [r.entity.get("id") for r in results[0]]
        return ids

    def done(self):
        self.collection.release()
        utility.drop_collection(self.collection_name)
        self.stop_milvus()
        disconnect_prometheus()

class MilvusFLAT(Milvus):
    def __init__(self, metric, dim, index_param, local, skip):
        super().__init__(metric, dim, index_param, local, skip)
        self.name = f"MilvusFLAT metric:{self._metric}"

    def get_index_param(self):
        return {
            "index_type": "FLAT",
            "metric_type": self._metric_type
        }

    def query(self, v, n):
        self.search_params = {
            "metric_type": self._metric_type,
        }
        results = self.collection.search(
            data = [v],
            anns_field = "vector",
            param = self.search_params,
            limit = n,
            output_fields=["id"]
        )
        ids = [r.entity.get("id") for r in results[0]]
        return ids

class MilvusIVFFLAT(Milvus):
    def __init__(self, metric, dim, index_param, local, skip):
        super().__init__(metric, dim, index_param, local, skip)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFFLAT metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"

class MilvusIVFSQ8(Milvus):
    def __init__(self, metric, dim, index_param, local, skip):
        super().__init__(metric, dim, index_param, local, skip)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_SQ8",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFSQ8 metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"

class MilvusIVFPQ(Milvus):
    def __init__(self, metric, dim, index_param, local, skip):
        super().__init__(metric, dim, index_param, local, skip)
        self._index_nlist = index_param.get("nlist", None)
        self._index_m = index_param.get("m", None)
        self._index_nbits = index_param.get("nbits", None)

    def get_index_param(self):
        assert self._dim % self._index_m == 0, "dimension must be able to be divided by m"
        return {
            "index_type": "IVF_PQ",
            "params": {
                "nlist": self._index_nlist,
                "m": self._index_m,
                "nbits": self._index_nbits if self._index_nbits else 8 
            },
            "metric_type": self._metric_type
        }
    
    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFPQ metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"

class MilvusHNSW(Milvus):
    def __init__(self, metric, dim, index_param, local=True, skip=False):
        super().__init__(metric, dim, index_param, local, skip)
        print("self.local: ", self.local)
        self._index_m = index_param.get("M", None)
        self._index_ef = index_param.get("efConstruction", None)
        print("self._index_ef: ", self._index_ef)

    def get_index_param(self):
        return {
            "index_type": "HNSW",
            "params": {
                "M": self._index_m,
                "efConstruction": self._index_ef
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, ef):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"ef": ef}
        }
        self.name = f"MilvusHNSW metric:{self._metric}, index_M:{self._index_m}, index_ef:{self._index_ef}, search_ef={ef}"

class MilvusSCANN(Milvus):
    def __init__(self, metric, dim, index_param, local, skip):
        super().__init__(metric, dim, index_param, local, skip)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "SCANN",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusSCANN metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"

INDICES = {
    "flat": {
        "constructor": MilvusFLAT,
        "args": None,
        "query_args": None,
    },
    "ivf_flat": {
        "constructor": MilvusIVFFLAT,
        "args": {
            "nlist": [128, 256, 512, 1024, 2048, 4096],
        },
        "query_args": [[1, 10, 20, 50, 100]],
    },
    "ivf_sq8": {
        "constructor": MilvusIVFSQ8,
        "args": {
            "nlist": [128, 256, 512, 1024, 2048, 4096],
        },
        "query_args": [[1, 10, 20, 50, 100]],
    },
    "ivf_pq": {
        "constructor": MilvusIVFPQ,
        "args": {
            "nlist": [128, 256, 512, 1024, 2048, 4096],
            "m": [2, 4],
        },
        "query_args": [[1, 10, 20, 50, 100]],
    }, 
    "hnsw": {
        "constructor": MilvusHNSW,
        "args": {
            "M": 4,
            # "M": [4, 8, 12, 16, 24, 36, 48, 64, 96],
            "efConstruction": 200,
            # "efConstruction": [200, 500],
        },
        "query_args": [10], # ef
        # "query_args": [[10, 20, 40, 80, 120, 200, 400, 600, 800]],
    },
    "scann": {
        "constructor": MilvusSCANN,
        "args": {
            "nlist": [64, 128, 256, 512, 1024, 2048, 4096, 8192],
        },
        "query_args": [[1, 10, 20, 30, 50]],
    }
}

def jaccard(a: List[int], b: List[int]) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0
    intersect = len(set(a) & set(b))
    return intersect / (float)(len(a) + len(b) - intersect)

def norm(a):
    return np.sum(a**2) ** 0.5

def euclidean(a, b):
    return norm(a - b)

class Metric(NamedTuple):
    distance: Callable[[np.ndarray, np.ndarray], float]
    distance_valid: Callable[[float], bool]

METRICS = {
    "hamming": Metric(
        distance=lambda a, b: np.mean(a.astype(np.bool_) ^ b.astype(np.bool_)),
        distance_valid=lambda a: True
    ),
    "jaccard": Metric(
        distance=lambda a, b: 1 - jaccard(a, b),
        distance_valid=lambda a: a < 1 - 1e-5
    ),
    "euclidean": Metric(
        distance=lambda a, b: euclidean(a, b),
        distance_valid=lambda a: True
    ),
    "angular": Metric(
        distance=lambda a, b: 1 - np.dot(a, b) / (norm(a) * norm(b)),
        distance_valid=lambda a: True
    ),
}

def convert_sparse_to_list(data: np.ndarray, lengths: List[int]) -> List[np.ndarray]:
    return [
        data[i - l : i] for i, l in zip(np.cumsum(lengths), lengths)
    ]

def dataset_transform(dataset: h5py.Dataset) -> Tuple[Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]]:
    if dataset.attrs.get("type", "dense") != "sparse":
        return np.array(dataset["train"]), np.array(dataset["test"])

    # we store the dataset as a list of integers, accompanied by a list of lengths in hdf5
    # so we transform it back to the format expected by the algorithms here (array of array of ints)
    return (
        convert_sparse_to_list(dataset["train"], dataset["size_train"]),
        convert_sparse_to_list(dataset["test"], dataset["size_test"])
    )

def build_index(algo: BaseANN, X_train: np.ndarray) -> Tuple:
    t0 = time.time()
    memory_usage_before = algo.get_memory_usage()
    algo.fit(X_train)
    build_time = time.time() - t0
    index_size = algo.get_memory_usage() - memory_usage_before # KB
    return build_time, index_size

def run_individual_query(algo: BaseANN, X_train: np.array, X_test: np.array, distance: str, count: int, run_count: int) -> Tuple[dict, list]:
    best_search_time = float("inf")
    for i in range(run_count):
        print("Run %d/%d..." % (i + 1, run_count))
        n_items_processed = [0]

        def single_query(v: np.array) -> Tuple[float, List[Tuple[int, float]]]:
            start = time.time()
            candidates = algo.query(v, count)
            total = time.time() - start

            assert len(candidates) == len(set(candidates)), "Implementation returned duplicated candidates"

            candidates = [
                (int(idx), float(METRICS[distance].distance(v, X_train[idx]))) for idx in candidates  # noqa
            ]
            n_items_processed[0] += 1

            if n_items_processed[0] % 1000 == 0:
                print("Processed %d/%d queries..." % (n_items_processed[0], len(X_test)))
            if len(candidates) > count:
                print(
                    "warning: algorithm %s returned %d results, but count"
                    " is only %d)" % (algo, len(candidates), count)
                )
            return (total, candidates)
    
        results = [single_query(x) for x in X_test]
        total_time = sum(time for time, _ in results)
        total_candidates = sum(len(candidates) for _, candidates in results)
        search_time = total_time / len(X_test)
        avg_candidates = total_candidates / len(X_test)
        best_search_time = min(best_search_time, search_time)

    verbose = hasattr(algo, "query_verbose")
    attrs = {
        "batch_mode": False,
        "best_search_time": best_search_time,
        "candidates": avg_candidates,
        "expect_extra": verbose,
        "name": str(algo),
        "run_count": run_count,
        "distance": distance,
        "count": int(count),
    }
    return (attrs, results)

def store_results(dataset: str, count: int, attrs, results):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "output")

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    filename = os.path.join(target_dir, dataset)
    print(f"store results in {filename}")

    with h5py.File(filename, "w") as f:
        for k, v in attrs.items():
            print(f"k: {k}, v: {v}")
            f.attrs[k] = v
        times = f.create_dataset("times", (len(results),), "f")
        neighbors = f.create_dataset("neighbors", (len(results), count), "i")
        distances = f.create_dataset("distances", (len(results), count), "f")
        
        for i, (t, ds) in enumerate(results):
            times[i] = t
            neighbors[i] = [n for n, d in ds] + [-1] * (count - len(ds))
            distances[i] = [d for n, d in ds] + [float("inf")] * (count - len(ds))

def run(a):
    start_time = time.time()
    print(a)
    D, dimension = get_dataset(a.dataset)
    print(D)
    X_train = np.array(D["train"])
    X_test = np.array(D["test"])

    print(f"Got a train set of size ({X_train.shape[0]} * {dimension})")
    print(f"Got {len(X_test)} queries")

    distance = D.attrs["distance"]
    print(f"distance: {distance}")
    X_train, X_test = dataset_transform(D)

    print(f"index: {a.index}")
    index = INDICES.get(a.index)
    print(index)
    if index is None:
        raise Exception("Not support index")
    
    args = index.get("args")
    query_args = index.get("query_args")

    algo = index.get("constructor")(metric=distance, dim=dimension, index_param=args, local=a.local, skip=a.skip)
 
    def sigint_handler(signum, frame):
        print("SIGINT captured ", frame)
        if algo.hasattr("p") and algo.p.pid() > 0:
            print("kill subprocess")
            algo.done()
            exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    build_index_time, index_size = build_index(algo, X_train)
    print(f"index size: {index_size}")

    query_start_time = time.time()

    if query_args:
        algo.set_query_arguments(*query_args)
    
    descriptor, results = run_individual_query(algo, X_train, X_test, distance, a.count, a.runs)
    query_end_time = time.time()
    print(f"query time: {query_end_time - query_start_time}")
    
    descriptor.update({
        "build_time": build_index_time,
        "index_size": index_size,
        "algo": algo,
        "dataset": a.dataset,
    })

    # store_results(a.dataset, a.count, descriptor, results)

    algo.done()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="sift-128-euclidean", choices=DATASETS.keys())
    parser.add_argument("-i", "--index", default="hnsw", choices=INDICES.keys())
    parser.add_argument("-k", "--count", default=10, help="the number of near neighbors to search for")
    parser.add_argument("-r", "--runs", default=1, help="run each algorithm instance")
    parser.add_argument("--local", action="store_true", default=True)
    parser.add_argument("--skip", action="store_true", default=False)
    args = parser.parse_args()
    run(args)
