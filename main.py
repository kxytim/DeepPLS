import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import torch as th
import torchvision
from torchvision import transforms
from models import DeepPLS


# load data
mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.resize_(28*28))])
training_dataset = torchvision.datasets.MNIST(root="./mnist", train=True, transform=mnist_transform)
test_dataset = torchvision.datasets.MNIST(root="./mnist", train=False, transform=mnist_transform)
training_data, training_label = training_dataset.data, training_dataset.targets[:, None]
test_data, test_label = test_dataset.data, test_dataset.targets[:, None]
N_samples = 1000
training_data, training_label = training_data[:N_samples], training_label[:N_samples]
label_real = test_label
training_data, test_data = 1/255*training_data, 1/255*test_data
training_data = training_data.reshape(-1, 28*28)
test_data = test_data.reshape(-1, 28*28)
training_label = preprocessing.OneHotEncoder().fit_transform(training_label).toarray()
test_label = preprocessing.OneHotEncoder().fit_transform(test_label).toarray()
training_label, test_label = th.Tensor(training_label), th.Tensor(test_label)

# fix random seed
np.random.seed(0)

# fit and predict
generalized_deep_pls = DeepPLS(
    lv_dimensions=[100, 150],
    pls_solver='svd',
    use_nonlinear_mapping=True,
    mapping_dimensions=[800, 800],
    nys_gamma_values=[0.014, 2.8],
    stack_previous_lv1=True
)

generalized_deep_pls.fit(training_data, training_label)
predictions = generalized_deep_pls.predict(test_data)
label_pred = th.argmax(predictions, dim=1, keepdim=True)
acc = accuracy_score(label_real, label_pred)
print('Classification Accuracy:', acc)
