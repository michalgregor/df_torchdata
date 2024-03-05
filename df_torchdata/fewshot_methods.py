from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch

#-----------------------------------
# SVM
#-----------------------------------

def train_func_svm(supervised_dataset, device=None, **kwargs):
    from sklearn.svm import SVC
    X, Y = supervised_dataset.cpu().to_tensor_dataset()

    kwargs = kwargs.copy()
    kwargs["kernel"] = kwargs.get("kernel", "linear")

    svc = SVC(**kwargs)
    svc.fit(X, Y)
    
    return lambda x: svc.predict(torch.as_tensor(x).cpu())

#-----------------------------------
# ProtoNet
#-----------------------------------

class ProtoNet(torch.nn.Module):
    def __init__(self, support_set, labels=None):
        super().__init__()
        self.labels = labels
        self.centers = self.compute_prototypes(support_set)

    def compute_prototypes(self, support_set):
        group_mean_dict = support_set.groupby().mean()

        if self.labels is None:
            self.labels = list(range(len(group_mean_dict)))

        group_means = [group_mean_dict[label] for label in self.labels]
        return torch.stack(group_means)

    def forward(self, X):
        dists = torch.cdist(X.float(), self.centers.float())
        probs = torch.softmax(-dists, dim=1)
        return probs
    
def train_func_protonet(fewshot_dataset, device=None, **kwargs):
    protonet = ProtoNet(fewshot_dataset, **kwargs).to(device)
    model = lambda x: protonet(x).argmax(dim=1)
    return model

#-----------------------------------
# KNN
#-----------------------------------

def train_func_knn(supervised_dataset, n_neighbors=3, device=None, **kwargs):
    from sklearn.neighbors import KNeighborsClassifier

    X, Y = supervised_dataset.cpu().to_tensor_dataset()

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
    knn.fit(X, Y)
    
    return lambda x: knn.predict(torch.as_tensor(x).cpu())

#-----------------------------------
# Eval functions
#-----------------------------------

def generalization_test(
    train_func, support_set, query_set, device=None,
    return_preds=True, print_summary=True, print_preds=True, **kwargs
):
    model = train_func(support_set, device=device, **kwargs)

    X_train, Y_train = support_set.to_tensor_dataset()
    X_train, Y_train = X_train, Y_train.cpu()

    X_test, Y_test = query_set.to_tensor_dataset()
    X_test, Y_test = X_test, Y_test.cpu()

    # on the training set
    preds_train = torch.as_tensor(model(X_train)).cpu()
    acc_train = accuracy_score(preds_train, Y_train)
    prec_train = precision_score(preds_train, Y_train, average='macro')
    rec_train = recall_score(preds_train, Y_train, average='macro')
    f1_train = f1_score(preds_train, Y_train, average='macro')

    if print_summary:
        print(
            f"\nResults @ train set: {(preds_train == Y_train).int().sum()}/{len(preds_train)} matches; "
            f"acc: {acc_train}; prec: {prec_train}; rec: {rec_train}; f1: {f1_train}"
        )

    if print_preds:
        print(f"Predictions on the train set: {preds_train}")
        print(f"True labels on the train set: {Y_train}")

    # on the test set
    preds_test = torch.as_tensor(model(X_test)).cpu()
    acc_test = accuracy_score(preds_test, Y_test)
    prec_test = precision_score(preds_test, Y_test, average='macro')
    rec_test = recall_score(preds_test, Y_test, average='macro')
    f1_test = f1_score(preds_test, Y_test, average='macro')
    
    if print_summary:
        print(
            f"\nResults @ test set (generalization): {(preds_test == Y_test).int().sum()}/{len(preds_test)} matches; "
            f"acc: {acc_test}; prec: {prec_test}; rec: {rec_test}; f1: {f1_test}"
        )

    if print_preds:
        print(f"Predictions on the test set: {preds_test}")
        print(f"True labels on the test set: {Y_test}")

    if return_preds:
        return acc_train, acc_test, preds_train, preds_test
    else:
        return acc_train, acc_test

def separability_test(
    train_func, support_set, query_set, device=None,
    return_preds=True, print_summary=True, print_preds=True, **kwargs
):
    full_dataset = support_set.concat(query_set)
    model = train_func(full_dataset, device=device, **kwargs)

    X, Y = full_dataset.to_tensor_dataset()
    preds = torch.as_tensor(model(X)).cpu()
    acc = accuracy_score(preds, Y)
    prec = precision_score(preds, Y, average='macro')
    rec = recall_score(preds, Y, average='macro')
    f1 = f1_score(preds, Y, average='macro')
    
    if print_summary:
        print(
            f"\nResults @ all data (separability): {(preds == Y).int().sum()}/{len(preds)} matches; "
            f"acc: {acc}; prec: {prec}; rec: {rec}; f1: {f1}"
        )

    if print_preds:
        print(f"Predictions on the all data: {preds}")
        print(f"True labels on the all data: {Y}")

    if return_preds:
        return acc, preds
    else:
        return acc
    
def svm_generalization_test(*args, **kwargs):
    return generalization_test(train_func_svm, *args, **kwargs)

def svm_separability_test(*args, **kwargs):
    return separability_test(train_func_svm, *args, **kwargs)
