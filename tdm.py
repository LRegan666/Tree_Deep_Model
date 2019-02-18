from sample_init import data_process, tree_generate_samples, merge_samples, Dataset
from deep_network import NeuralNet
from prediction import metrics_count
from construct_tree import TreeLearning


def main():
    data_train, data_val, data_test, cache = data_process()
    _, _, _, tree = cache
    item_ids, item_size = tree.items, len(tree.items)
    model = None
    num_epoch = 20
    while num_epoch > 0:
        tree_samples = tree_generate_samples(item_ids, tree.leaf_dict, tree.root)
        tdata, vdata = merge_samples(data_train, tree_samples), merge_samples(data_val, tree_samples)
        dtrain, vtrain = Dataset(tdata, 50, shuffle=True), Dataset(vdata, 50)
        vtest = Dataset(data_val, 50)
        model = NeuralNet(item_size, tree.node_size, 24)
        model.train(use_gpu=True,
                    train_data=dtrain,
                    validate_data=vtrain)
        metrics_count(vtest, tree.root, 10, model)
        num_epoch -= 1
        if num_epoch > 0:
            item_embeddings = model.get_embeddings(item_ids)
            tree = TreeLearning(item_embeddings, item_ids)
            _ = tree.clustering_binary_tree()
    dtest = Dataset(data_test, 100)
    metrics_count(dtest, tree.root, 150, model)
    print("========================================== end ==========================================")


if __name__ == '__main__':
    main()

