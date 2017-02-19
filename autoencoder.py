import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset

import sys

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

N_EPOCH = 100

def plot_mnist_data(samples):
    for index, (data, label) in enumerate(samples):
        plt.subplot(4, 4, index + 1)
        plt.axis('off')
        plt.imshow(data.reshape(28, 28), cmap=cm.gray_r, interpolation='nearest')
        n = int(label)
        plt.title(n, color='red')
    plt.savefig("./pict/epoch_"+str(N_EPOCH)+'.png')
    #plt.show()

class Autoencoder(chainer.Chain):
    def __init__(self):
        super(Autoencoder, self).__init__(
                encoder = L.Linear(784, 64),
                decoder = L.Linear(64, 784))

    def __call__(self, x, hidden=False):
        h = F.relu(self.encoder(x))
        if hidden:
            return h
        else:
             return F.relu(self.decoder(h))

train, test = chainer.datasets.get_mnist()
train = train[0:1000]
test = test[47:63]
#plot_mnist_data(test)

train = [i[0] for i in train]
train = tuple_dataset.TupleDataset(train, train)
train_iter = chainer.iterators.SerialIterator(train, 100)

model = L.Classifier(Autoencoder(), lossfun=F.mean_squared_error)
model.compute_accuracy = False
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (N_EPOCH, 'epoch'), out="result")
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport( ['epoch', 'main/loss']))
trainer.extend(extensions.ProgressBar())

trainer.run()

pred_list = []
for (data, label) in test:
    pred_data = model.predictor(np.array([data]).astype(np.float32)).data
    pred_list.append((pred_data, label))
plot_mnist_data(pred_list)
