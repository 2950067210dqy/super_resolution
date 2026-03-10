from megaflow.dataset.MegaFlow2D import MegaFlow2D

if __name__ == '__main__':
    dataset = MegaFlow2D(root='/dataset', download=True, transform='normalize', pre_transform=None, split_scheme='mixed', split_ratio=0.8)
    # if the dataset is not processed, the process function will be called automatically.
    # to facilitate multi-thread processing, be sure to exceute the process function in '__main__'.

    # get one sample
    sample_low, sample_high = dataset.get(0)
    print('Number of nodes: {}, number of edges: {}'.format(sample_low.num_nodes, sample_low.num_edges))