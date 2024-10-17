from .KVClusterV2 import KVClusterV2

def clustering_inference(iteration):
    cluster = KVClusterV2(10)
    cluster.load_model('/home/jiaxi/cs525/Experiment/prediction_model/clustering/train_result.json')

    sample = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    for i in range(iteration):
        cluster.tell(sample)