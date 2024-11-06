from datasetloader.particle import FluidflowDataset3D

def build_train_dataset(dataset):
    if dataset == 'particle':
        train_dataset = FluidflowDataset3D(npoints=2048, root='TR-Flow3D/Train', partition='train')
    else:
        raise ValueError(f'stage {dataset} is not supported')

    return train_dataset

def build_test_dataset(dataset):
    if dataset == 'particle':
        test_dataset = FluidflowDataset3D(npoints=2048, root='TR-Flow3D/TEST', partition='test') 
    else:
        raise ValueError(f'stage {dataset} is not supported')

    return test_dataset



