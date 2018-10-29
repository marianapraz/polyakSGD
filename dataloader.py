from torchvision.datasets import MNIST
import torchnet as tnt
import os
import torchvision.transforms as transforms

def mnist(bs,train=True):
    ds = MNIST(root=os.path.join('data/','mnist'), download=True, train=train)

    data = getattr(ds, 'train_data' if train else 'test_data')
    labels = getattr(ds, 'train_labels' if train else 'test_labels')

    kwargs = {'num_workers': 0,'drop_last': True}

    tds = tnt.dataset.TensorDataset([data, labels])
    augment = tnt.transform.compose([lambda x: x[None,:,:].float() /255.0])
    tds = tds.transform({0:augment})

    tds = tds.parallel(batch_size=bs, shuffle=True, **kwargs)

    #print(tds.size)
    return tds