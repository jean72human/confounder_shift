# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import itertools
import copy
import numpy as np
from collections import OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from sklearn.metrics import confusion_matrix

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, l2_between_dicts, proj
)

def check_gradient_similarity(net1, net2, threshold=1e-6):
    for (name1, p1), (name2, p2) in zip(net1.named_parameters(), net2.named_parameters()):
        if p1.grad is None or p2.grad is None:
            print(f"No gradient for {name1}")
            continue
            
        diff = torch.abs(p1.grad - p2.grad).max().item()
        if diff > threshold:
            print(f"Gradient mismatch in {name1}: max difference = {diff}")
            return False
    return True

ALGORITHMS = [
    'ERM',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'CAD',
    'CondCAD',
    'Transfer',
    'CausIRL_CORAL',
    'CausIRL_MMD',
    'FLR',
    'LLR',
    'CBFT',
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

import torch.nn as nn
import torch.nn.init as init
# import networks  # Assuming networks is a custom module containing your Classifier class

def xavier_uniform_init(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)

def kaiming_normal_init(module):
    if isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)

def zero_init(module):
    if isinstance(module, nn.Linear):
        module.weight.data.fill_(0.0)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    
    def predict(self, x):
        return self.network(x)


class ERM_WL(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_WL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y, z in minibatches])
        all_y = torch.cat([y for x, y, z in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)



class GroupDRO_WL(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO_WL, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())
        self.n_u = 2

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        all_x = torch.cat([x for x, y, u in minibatches])
        all_y = torch.cat([y for x, y, u in minibatches])
        all_u = torch.cat([u for x, y, u in minibatches])

        if not len(self.q):
            self.q = torch.ones(self.n_u).to(device)

        losses = torch.zeros(self.n_u).to(device)

        for m in range(self.n_u):
            idx = (all_u == m)
            x = all_x[idx]
            y = all_y[idx]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}



class SUBG(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SUBG, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.n_u = 2
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        device = minibatches[0][0].device
        all_x = torch.cat([x for x, y, u in minibatches]).to(device)
        all_y = torch.cat([y for x, y, u in minibatches]).to(device)
        all_u = torch.cat([u for x, y, u in minibatches]).to(device)

        # Subsample to balance the number of examples for each (y, u) group
        unique_u = torch.unique(all_u)
        unique_y = torch.unique(all_y)
        balanced_x = []
        balanced_y = []
        balanced_u = []

        min_samples = float('inf')
        for u_class in unique_u:
            for y_class in unique_y:
                count = ((all_u == u_class) & (all_y == y_class)).sum().item()
                if count < min_samples:
                    min_samples = count

        for u_class in unique_u:
            for y_class in unique_y:
                indices = ((all_u == u_class) & (all_y == y_class)).nonzero(as_tuple=True)[0]
                subsample_indices = indices[torch.randperm(len(indices))[:min_samples]]
                balanced_x.append(all_x[subsample_indices])
                balanced_y.append(all_y[subsample_indices])
                balanced_u.append(all_u[subsample_indices])

        all_x = torch.cat(balanced_x)
        all_y = torch.cat(balanced_y)
        all_u = torch.cat(balanced_u)

        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class UShift2(Algorithm):
    """
    Empirical Risk Minimization (ERM) with extra logic to keep/use the best models.

    Each model in self.networks[i] deals only with data for which u == i.
    So, when we evaluate self.networks[i], we only look at x,y with u == i.
    In contrast, self.network_u is for classifying u across all data.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(UShift2, self).__init__(input_shape, num_classes, num_domains,
                                      hparams)
        self.n_u = 2
        self.momentum = 0.1
        self.w = torch.zeros((self.n_u, 1))
        self.C = torch.zeros((self.n_u, self.n_u))
        self.num_classes = num_classes

        # Build main networks (one for each possible value of u) + network_u for classifying u
        self.networks = nn.ModuleList()
        for _ in range(self.n_u):
            featurizer = networks.Featurizer(input_shape, self.hparams)
            classifier = networks.Classifier(
                featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier']
            )
            self.networks.append(nn.Sequential(featurizer, classifier))

        featurizer_u = networks.Featurizer(input_shape, self.hparams)
        classifier_u = networks.Classifier(
            featurizer_u.n_outputs,
            self.n_u,
            self.hparams['nonlinear_classifier']
        )
        self.network_u = nn.Sequential(featurizer_u, classifier_u)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            itertools.chain(
                *[net.parameters() for net in (list(self.networks) + [self.network_u])]
            ),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        ######################################################################
        # Initialize "best copies" of each model + their accuracies
        ######################################################################
        self.best_networks = nn.ModuleList()
        self.best_networks_accuracies = []
        for net in self.networks:
            net_copy = copy.deepcopy(net)
            self.best_networks.append(net_copy)
            self.best_networks_accuracies.append(0.0)

        self.best_network_u = copy.deepcopy(self.network_u)
        self.best_network_u_accuracy = 0.0

    def update(self, minibatches, loader, unlabeled=None, step=0):
        """
        Perform one training step using `minibatches`.
        Only every 100 steps do we check accuracy on the `loader`
        and potentially update the best model copies.

        minibatches: list of (x, y, u) tuples
        loader: DataLoader for evaluation
        unlabeled: (unused here)
        step: current step number (for the 100-step check)
        """
        device = minibatches[0][0].device

        # 1) TRAINING STEP with minibatches
        # ------------------------------------------------------
        # Gather all x, y, u from minibatches
        all_x = torch.cat([x for x, y, u in minibatches]).to(device)
        all_y = torch.cat([y for x, y, u in minibatches]).to(device)
        all_u = torch.cat([u for x, y, u in minibatches]).to(device)

        # Subsample to balance the number of examples for each (y, u) group
        unique_u_vals = torch.unique(all_u)
        unique_y_vals = torch.unique(all_y)
        balanced_x = []
        balanced_y = []
        balanced_u = []

        min_samples = float('inf')
        for u_class in unique_u_vals:
            for y_class in unique_y_vals:
                count = ((all_u == u_class) & (all_y == y_class)).sum().item()
                if count < min_samples:
                    min_samples = count

        for u_class in unique_u_vals:
            for y_class in unique_y_vals:
                idxs = ((all_u == u_class) & (all_y == y_class)).nonzero(as_tuple=True)[0]
                selected = idxs[torch.randperm(len(idxs))[:min_samples]]
                balanced_x.append(all_x[selected])
                balanced_y.append(all_y[selected])
                balanced_u.append(all_u[selected])

        all_x = torch.cat(balanced_x)
        all_y = torch.cat(balanced_y)
        all_u = torch.cat(balanced_u)

        # Recreate confusion matrix C
        self.create_C(all_x, all_u)

        # Forward pass & compute losses
        pred_u = self.network_u(all_x)
        loss = F.cross_entropy(pred_u, all_u)

        # For classification of y: each self.networks[i] handles data when u==i
        # But we do a "mixture" approach (like in your original code):
        ys = torch.cat([net(all_x).unsqueeze(-1) for net in self.networks], dim=-1)
        # shape: (N, num_classes, n_u)
        pred_y = torch.bmm(
            ys,
            F.one_hot(all_u, num_classes=self.n_u).float().unsqueeze(-1)
        ).squeeze()  # => shape: (N, num_classes)
        loss += F.cross_entropy(pred_y, all_y)

        # Backprop and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 2) OPTIONAL EVAL every 100 steps
        # ------------------------------------------------------
        if step % 100 == 0:
            # Evaluate each network i only on data for which u == i
            for i, net in enumerate(self.networks):
                current_acc = self._eval_accuracy_on_loader_group_balance(
                    net, loader, device, is_u=False, network_idx=i
                )
                if current_acc > self.best_networks_accuracies[i]:
                    self.best_networks_accuracies[i] = current_acc
                    self.best_networks[i].load_state_dict(
                        copy.deepcopy(net.state_dict())
                    )

            # Evaluate network_u (classification of u) on all data
            current_acc_u = self._eval_accuracy_on_loader_group_balance(
                self.network_u, loader, device, is_u=True, network_idx=None
            )
            if current_acc_u > self.best_network_u_accuracy:
                self.best_network_u_accuracy = current_acc_u
                self.best_network_u.load_state_dict(
                    copy.deepcopy(self.network_u.state_dict())
                )

        return {'loss': loss.item()}

    def create_w(self, adapt_x):
        device = adapt_x.device
        mu = torch.zeros(self.n_u, 1, device=device)

        u = self.network_u(adapt_x)
        _, u = torch.max(u.unsqueeze(-1), dim=1)
        for i in range(self.n_u):
            mu[i, 0] = (u == i).sum().item()
        mu = mu / (mu.sum() + 1e-8)

        w = torch.linalg.pinv(self.C) @ mu
        w = torch.clip(w, min=1e-2, max=15)
        w = w / (w.sum() + 1e-8)
        self.w = w.to(device)
    
    def create_C(self, all_x, all_u):
        device = all_x.device
        u = self.network_u(all_x)
        _, u = torch.max(u.unsqueeze(-1), dim=1)
        cm = confusion_matrix(all_u.cpu(), u.cpu(), labels=list(range(self.n_u)), normalize='true')
        new_C = torch.from_numpy(np.array(cm)).float().T.to(device)
        self.C = new_C

    def predict(self, x, u=None):
        """
        We use the best copies, not the current/active versions.
        """
        if u is None:
            # We first guess u using self.best_network_u, then
            # pick each self.best_networks[i] accordingly.
            self.create_w(x)
            pred_u = self.best_network_u(x).softmax(-1).unsqueeze(-1)
            ys = torch.cat([
                net(x).softmax(-1).unsqueeze(-1) for net in self.best_networks
            ], dim=-1)
            pred_y = torch.bmm(
                ys,
                pred_u * self.w[None, ...].to(x.device)
            ).squeeze()
        else:
            # If user provides the value of u, we do a simpler combination
            pred_u = F.one_hot(u, num_classes=self.n_u).float().unsqueeze(-1).to(x.device)
            ys = torch.cat([
                net(x).softmax(-1).unsqueeze(-1) for net in self.best_networks
            ], dim=-1)
            pred_y = torch.bmm(ys, pred_u).squeeze()

        return pred_y

    ######################################################################
    # Evaluate accuracy on loader with group balancing
    ######################################################################
    def _eval_accuracy_on_loader_group_balance(
        self, model, loader, device, is_u=False, network_idx=None
    ):
        """
        If is_u == True: measure accuracy on *all* data for classifying u.
          - Group-balance among (y, u).
        
        If is_u == False: measure accuracy on the subset of data for which u == network_idx,
          then group-balance among y.
        """
        model.eval()

        all_x = []
        all_y = []
        all_u = []

        # 1) Collect full data from loader
        with torch.no_grad():
            for x, y, u in loader:
                all_x.append(x)
                all_y.append(y)
                all_u.append(u)

        if len(all_x) == 0:
            # Loader might be empty; avoid errors
            model.train()
            return 0.0

        all_x = torch.cat(all_x).to(device)
        all_y = torch.cat(all_y).to(device)
        all_u = torch.cat(all_u).to(device)

        if is_u:
            # -----------------------------------------------
            # EVALUATE network_u ACROSS ALL (y, u) DATA
            # Group-balance among (y, u)
            # -----------------------------------------------
            unique_y_vals = torch.unique(all_y)
            unique_u_vals = torch.unique(all_u)

            min_samples = float('inf')
            for yval in unique_y_vals:
                for uval in unique_u_vals:
                    count = ((all_y == yval) & (all_u == uval)).sum().item()
                    if count < min_samples:
                        min_samples = count

            if min_samples == 0:
                # If there's any (y, u) pair with no samples, we can skip or default accuracy
                model.train()
                return 0.0

            balanced_x = []
            balanced_y = []
            balanced_u = []
            for yval in unique_y_vals:
                for uval in unique_u_vals:
                    indices = ((all_y == yval) & (all_u == uval)).nonzero(as_tuple=True)[0]
                    selected = indices[torch.randperm(len(indices))[:min_samples]]
                    balanced_x.append(all_x[selected])
                    balanced_y.append(all_y[selected])
                    balanced_u.append(all_u[selected])

            balanced_x = torch.cat(balanced_x)
            balanced_u = torch.cat(balanced_u)

            outputs = model(balanced_x)  # shape: (N, n_u)
            preds = outputs.argmax(dim=1)
            correct = (preds == balanced_u).sum().item()
            total = balanced_u.size(0)
            acc = correct / total if total > 0 else 0.0

        else:
            # -----------------------------------------------
            # EVALUATE networks[i] ONLY ON DATA WHERE u == i
            # Group-balance among y
            # -----------------------------------------------
            assert network_idx is not None, (
                "network_idx must be provided when is_u=False"
            )

            # Filter out data for which u != network_idx
            mask = (all_u == network_idx)
            filtered_x = all_x[mask]
            filtered_y = all_y[mask]

            if filtered_x.size(0) == 0:
                # No data for that u in the loader
                model.train()
                return 0.0

            # Now group balance among y
            unique_y_vals = torch.unique(filtered_y)
            min_samples = float('inf')
            # find smallest group across y classes
            for yval in unique_y_vals:
                count = (filtered_y == yval).sum().item()
                if count < min_samples:
                    min_samples = count

            if min_samples == 0:
                model.train()
                return 0.0

            balanced_x = []
            balanced_y = []
            for yval in unique_y_vals:
                indices = (filtered_y == yval).nonzero(as_tuple=True)[0]
                selected = indices[torch.randperm(len(indices))[:min_samples]]
                balanced_x.append(filtered_x[selected])
                balanced_y.append(filtered_y[selected])

            balanced_x = torch.cat(balanced_x)
            balanced_y = torch.cat(balanced_y)

            outputs = model(balanced_x)  # shape: (N, num_classes)
            preds = outputs.argmax(dim=1)
            correct = (preds == balanced_y).sum().item()
            total = balanced_y.size(0)
            acc = correct / total if total > 0 else 0.0

        model.train()
        return acc


# class UShift2(Algorithm):
#     """
#     Empirical Risk Minimization (ERM)
#     """

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(UShift2, self).__init__(input_shape, num_classes, num_domains,
#                                   hparams)
#         self.n_u = 2
#         self.momentum = 0.1
#         self.w = torch.zeros((self.n_u, 1))
#         self.C = torch.zeros((self.n_u,self.n_u))
#         self.num_classes = num_classes
#         self.temp_path = "cmnist_output_ushift/ushift"
#         self.best_acc = [0] * self.n_u
#         self.best_acc_u = 0

#         self.networks = nn.ModuleList()
#         for _ in range(self.n_u):
#             featurizer = networks.Featurizer(input_shape, self.hparams)
#             classifier = networks.Classifier(
#                 featurizer.n_outputs,
#                 num_classes,
#                 self.hparams['nonlinear_classifier'])

#             self.networks.append(nn.Sequential(featurizer,classifier))

#         featurizer_u = networks.Featurizer(input_shape, self.hparams)
#         classifier_u = networks.Classifier(
#             featurizer_u.n_outputs,
#             self.n_u,
#             self.hparams['nonlinear_classifier'])

#         self.network_u = nn.Sequential(featurizer_u,classifier_u)

#         self.optimizer = torch.optim.Adam(
#             itertools.chain(*[network.parameters() for network in self.networks+[self.network_u]]),
#             lr=self.hparams["lr"],
#             weight_decay=self.hparams['weight_decay']
#         )

#     def train_submodels(self, train_minibatches_iterator, uda_minibatches_iterator, eval_loader, n_steps, device):
#         for step in range(n_steps):
#             minibatches = [(x.to(device), y.to(device), u.to(device)) for x,y,u in next(train_minibatches_iterator)]
#             all_x = torch.cat([x for x, y, u in minibatches]).to(device)
#             all_y = torch.cat([y for x, y, u in minibatches]).to(device)
#             all_u = torch.cat([u for x, y, u in minibatches]).to(device)
#             submodel_accus = []
#             for idx,network in enumerate(self.networks):
#                 x = all_x[all_u==idx]
#                 y = all_y[all_u==idx]

#                 # Subsample to balance classes within each group
#                 unique_y = torch.unique(y)
#                 balanced_x = []
#                 balanced_y = []
#                 min_samples = float('inf')
#                 for y_class in unique_y:
#                     count = (y == y_class).sum().item()
#                     if count < min_samples:
#                         min_samples = count

#                 for y_class in unique_y:
#                     indices = (y == y_class).nonzero(as_tuple=True)[0]
#                     subsample_indices = indices[torch.randperm(len(indices))[:min_samples]]
#                     balanced_x.append(x[subsample_indices])
#                     balanced_y.append(y[subsample_indices])

#                 x = torch.cat(balanced_x)
#                 y = torch.cat(balanced_y)

#                 self._train_submodel(network,x,y)
#                 accu = self._eval_submodel(network, eval_loader, idx)
#                 submodel_accus.append(accu)
#             u_accu = self._eval_submodel_u(self.network_u, eval_loader)
#             self._train_submodel(self.network_u,all_x,all_u)
#             if (step + 1) % 100 == 0:
#                 print(f"Step {step+1}:")
#                 for i, accu in enumerate(submodel_accus):
#                     print(f"\tAccuracy of submodel {i}: {accu}")
#                 print(f"\tAccuracy of submodel u: {u_accu}")
#         self.load_best()
#         self.create_C(all_x,all_u)
#         self.create_w(all_x)

#     def _train_submodel(self, network, x, y):
#         """
#         Performs a single step of gradient descent on the given network using the provided data.
#         """
#         loss = F.cross_entropy(network(x), y)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#     def _eval_submodel(self, network, eval_loader, idx):
#         """
#         Evaluates the given network on the data in eval_loader where u=idx and saves the model if it's the best so far.
#         Subsamples the data to balance it class-wise within each group.
#         """
#         correct = 0
#         total = 0
#         network.eval()
#         with torch.no_grad():
#             all_x, all_y, all_u = [], [], []
#             for x, y, u in eval_loader:
#                 x, y, u = x.to(next(network.parameters()).device), y.to(next(network.parameters()).device), u.to(next(network.parameters()).device)
#                 mask = (u == idx)
#                 if mask.sum() > 0:
#                     all_x.append(x[mask])
#                     all_y.append(y[mask])
#                     all_u.append(u[mask])

#             if not all_x:  # No data for this group
#                 return 0

#             all_x = torch.cat(all_x)
#             all_y = torch.cat(all_y)
#             all_u = torch.cat(all_u)

#             # # Subsample to balance classes within each group
#             unique_y = torch.unique(all_y)
#             balanced_x = []
#             balanced_y = []
#             min_samples = float('inf')
#             for y_class in unique_y:
#                 count = (all_y == y_class).sum().item()
#                 if count < min_samples:
#                     min_samples = count

#             for y_class in unique_y:
#                 indices = (all_y == y_class).nonzero(as_tuple=True)[0]
#                 subsample_indices = indices[torch.randperm(len(indices))[:min_samples]]
#                 balanced_x.append(all_x[subsample_indices])
#                 balanced_y.append(all_y[subsample_indices])

#             all_x = torch.cat(balanced_x)
#             all_y = torch.cat(balanced_y)

#             outputs = network(all_x)
#             _, predicted = torch.max(outputs.data, 1)
#             total += all_y.size(0)
#             correct += (predicted == all_y).sum().item()

#         acc = correct / total if total > 0 else 0
#         network.train()
#         if acc > self.best_acc[idx]:
#             self.best_acc[idx] = acc
#             torch.save(network.state_dict(), f"{self.temp_path}_{idx}.pth")
#         return acc

#     def _eval_submodel_u(self, network, eval_loader):
#         """
#         Evaluates the given network on the data in eval_loader where u=idx and saves the model if it's the best so far.
#         """
#         correct = 0
#         total = 0
#         network.eval()
#         with torch.no_grad():
#             for x, y, u in eval_loader:
#                 x, u = x.to(next(network.parameters()).device), u.to(next(network.parameters()).device)

#                 outputs = network(x)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += u.size(0)
#                 correct += (predicted == u).sum().item()

#         acc = correct / total if total > 0 else 0
#         network.train()
#         if acc > self.best_acc_u:
#             self.best_acc_u = acc
#             torch.save(network.state_dict(), f"{self.temp_path}_u.pth")
#         return acc

#     def create_w(self, adapt_x):
#         device = adapt_x.device
#         mu = torch.zeros(self.n_u,1).to(device)
#         matrices = []

#         # features = self.featurizer(adapt_x)
#         # u = self.classifieru(features)
#         u = self.network_u(adapt_x)
#         _, u = torch.max(u.unsqueeze(-1),1)
#         for i in range(self.n_u):
#             mu[i,0] = (u==i).sum().item()
#         # print(mu)
#         mu = mu/mu.sum()
#         # print(mu)

#         w = torch.linalg.pinv(self.C) @ mu
#         w = torch.clip(w, min=1e-2, max=15)
#         w = w / w.sum()
#         self.w = w.to(adapt_x.device)
    
#     def create_C(self, all_x, all_u):
#         device = all_x.device
#         # features = self.featurizer(all_x)
#         # u = self.classifiery(features)
#         u = self.network_u(all_x)
#         _, u = torch.max(u.unsqueeze(-1),1)
#         cm = confusion_matrix(all_u.cpu(), u.cpu(), labels=list(range(self.n_u)), normalize='true')
#         new_C = torch.from_numpy(np.array(cm)).float().T.to(device)
        
#         self.C = new_C.to(all_x.device)

#     def load_best(self):
#         for idx,network in enumerate(self.networks):
#             network.load_state_dict(torch.load(f"{self.temp_path}_{idx}.pth"))
#         self.network_u.load_state_dict(torch.load(f"{self.temp_path}_u.pth"))

#     def predict(self, x, u=None):
#         if u is None:
#             self.create_w(x)
#             pred_u = self.network_u(x).softmax(-1).unsqueeze(-1)
#             ys = torch.cat([network(x).softmax(-1).unsqueeze(-1) for network in self.networks],-1)
#             pred_y = torch.bmm(ys,pred_u*self.w[None,...].to(x.device)).squeeze()
#         else:
#             pred_u = F.one_hot(u,num_classes=self.n_u).float().unsqueeze(-1).to(x.device)
#             ys = torch.cat([network(x).softmax(-1).unsqueeze(-1) for network in self.networks],-1)
#             pred_y = torch.bmm(ys,pred_u).squeeze()

#         return pred_y




class FLR(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(FLR, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None, retrain=False):
        # minibatch and unlabeled are the same type of object
        if retrain:
            data_batches = unlabeled
        else:
            data_batches = minibatches
        all_x = torch.cat([x for x, y in data_batches])
        all_y = torch.cat([y for x, y in data_batches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class LLR(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(LLR, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.classes = num_classes

    def update(self, minibatches, unlabeled=None, retrain=False, reinit=False, initialization_method='xavier_uniform'):
        # minibatch and unlabeled are the same type of object
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        data_batches = minibatches if not retrain else unlabeled
        all_x = torch.cat([x for x, y in data_batches])
        all_y = torch.cat([y for x, y in data_batches])
        if reinit:
            self.classifier = networks.Classifier(
                self.featurizer.n_outputs,
                self.classes,
                self.hparams['nonlinear_classifier'])
            self.classifier.to(device)
            if initialization_method == 'xavier_uniform':
                self.classifier.apply(xavier_uniform_init)
            elif initialization_method == 'kaiming_normal':
                self.classifier.apply(kaiming_normal_init)
            elif initialization_method == 'zero':
                self.classifier.apply(zero_init)
            else:
                raise ValueError("Invalid initialization method.")
            self.network = nn.Sequential(self.featurizer, self.classifier)
        if retrain:
            for param in self.featurizer.parameters():
                param.requires_grad = False
            all_features = self.featurizer(all_x)
            loss = F.cross_entropy(self.classifier(all_features), all_y)
        else:
            loss = F.cross_entropy(self.predict(all_x), all_y)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)
    
class CBFT(Algorithm):
    """
    Connectivity Based Fine Tuning https://arxiv.org/abs/2211.08422
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CBFT, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(self.featurizer.n_outputs, num_classes, self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])
        self.criterion = nn.CrossEntropyLoss()
        self.lambd = hparams.get("lambd", 1)
        self.warmup_epochs = hparams.get("warmup_epochs", 1)
        self.loss_margin = hparams.get("loss_margin", 1.0)
        self.interpolated_network = None  # The interpolated network should be initialized.
        self.classes = num_classes
        self.input_shape = input_shape

    # TODO: implement all of this
    def update(self, minibatches, unlabeled=None, epoch=10, retrain=False, reinit=False, initialization_method='xavier_uniform'):
        # minibatch and unlabeled are the same type of object
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        if reinit:
            # self.classifier = networks.Classifier(
            #     self.featurizer.n_outputs,
            #     self.classes,
            #     self.hparams['nonlinear_classifier'])
            # self.classifier.to(device)
            # if initialization_method == 'xavier_uniform':
            #     self.classifier.apply(xavier_uniform_init)
            # elif initialization_method == 'kaiming_normal':
            #     self.classifier.apply(kaiming_normal_init)
            # elif initialization_method == 'zero':
            #     self.classifier.apply(zero_init)

            # create a new linear layer with the same number of outputs as the old classifier, and copy the weights
            self.old_classifier = networks.Classifier(
                self.featurizer.n_outputs,
                self.classes,
                self.hparams['nonlinear_classifier'])
            self.old_classifier.to(device)
            self.old_classifier.load_state_dict(self.classifier.state_dict())

            # create a new feaurizer with the same number of outputs as the old featurizer, and copy the weights
            self.old_featurizer = networks.Featurizer(self.input_shape, self.hparams)
            self.old_featurizer.to(device)
            self.old_featurizer.load_state_dict(self.featurizer.state_dict())

            self.old_featurizer.to(device)
            self.old_network = nn.Sequential(self.old_featurizer, self.old_classifier)
            self.old_network.to(device)
            self.interpolated_network = copy.deepcopy(self.old_network)
            self.interpolated_network.to(device)
            self.old_network.eval()
            self.interpolated_network.train()
            # else:
            #     raise ValueError("Invalid initialization method.")
            self.network = nn.Sequential(self.featurizer, self.classifier)
        if retrain and unlabeled is not None and self.interpolated_network is not None:

            new_data_batches = unlabeled
            old_data_batches = minibatches

            all_new_x = torch.cat([x for x, y in new_data_batches])
            all_new_y = torch.cat([y for x, y in new_data_batches])

            all_old_x = torch.cat([x for x, y in old_data_batches])
            all_old_y = torch.cat([y for x, y in old_data_batches])

            all_features_new_data = self.featurizer(all_new_x)
            all_features_old_data = self.featurizer(all_old_x)

            n_classes = self.classifier.weight.shape[0]
            class_ids_new = {i: (all_new_y == i).nonzero().view(-1,1).type(torch.long) for i in range(n_classes)} 
            class_ids_old = {i: (all_old_x == i).nonzero().view(-1,1).type(torch.long) for i in range(n_classes)}
            

            with torch.no_grad():
                t = 0.5 + 0.5 * (torch.rand(1)).clip(-0.5, 0.5)[0]
                for module_old, module_new, module_interpolated in zip(self.old_network.modules(), self.network.modules(), self.interpolated_network.modules()):
                    if hasattr(module_old, 'weight') and hasattr(module_new, 'weight') and hasattr(module_interpolated, 'weight'):
                        module_interpolated.weight.data = t * module_old.weight.data + (1 - t) * module_new.weight.data
                    if hasattr(module_old, 'bias') and hasattr(module_new, 'bias') and hasattr(module_interpolated, 'bias') and module_interpolated.bias is not None:
                        module_interpolated.bias.data = t * module_old.bias.data + (1 - t) * module_new.bias.data
                    
            # Get barrier loss: compute loss and backward in interpolated network
            self.interpolated_network.zero_grad()
            self.network.zero_grad()
            outputs_interpolated = self.interpolated_network(all_old_x)
            loss = (self.loss_margin - F.cross_entropy(outputs_interpolated, all_old_y)).abs()
            loss.backward()

            with torch.no_grad():
                for module_new, module_interpolated in zip(self.network.modules(), self.interpolated_network.modules()):
                    if hasattr(module_interpolated, 'weight') and hasattr(module_interpolated.weight, 'grad'):
                        module_new.weight.grad = (1-t) * module_interpolated.weight.grad
                    if hasattr(module_interpolated, 'bias') and module_interpolated.bias is not None and hasattr(module_interpolated.bias, 'grad'):
                        module_new.bias.grad = (1-t) * module_interpolated.bias.grad

            # Update network with barrier loss
            self.optimizer.step()

            # Get invariance loss and fine tuning loss
            self.network.zero_grad()
            if epoch >= self.warmup_epochs:
                loss = F.cross_entropy(self.network(all_new_x), all_new_y)
            else:
                inv_loss = 0
                for i in range(n_classes):
                    if (class_ids_new[i].shape[0] == 0 or class_ids_old[i].shape[0] == 0):
                        continue
                    inv_loss += (F.normalize(all_features_new_data[class_ids_new[i][:,0]].mean(dim=0, keepdim=True), dim=1) - F.normalize(all_features_old_data[class_ids_old[i][:,0]].mean(dim=0, keepdim=True), dim=1)).norm().pow(2)
                # invariance loss
                loss = inv_loss / n_classes
                # fine tuning loss
                loss += F.cross_entropy(self.network(all_new_x), all_new_y)
            
            # Update network with fine tuning loss and invariance loss
            loss.backward()
            self.optimizer.step()

            if epoch >= self.warmup_epochs:
                # self.lr_scheduler.step()
                # TODO: sort our learning rate scheduler
                pass

            torch.cuda.empty_cache()

        else:
            data_batches = minibatches
            all_x = torch.cat([x for x, y in data_batches])
            all_y = torch.cat([y for x, y in data_batches])
            loss = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


    def old_update(self, minibatches, unlabeled=None, epoch=0, retrain=False, reinit=False, initialization_method='xavier_uniform'):
        self.network.train()
        minibatches_nc, minibatches_c = minibatches
        device = "cuda" if minibatches_nc[0][0].is_cuda else "cpu"
        train_loss_c, train_loss_nc, correct_c, correct_nc, total_c, total_nc = 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8
        n_classes = self.network.linear.weight.shape[0] # Assuming the network has a linear layer

        for batch_idx, ((inputs_nc, targets_nc), (inputs_c, targets_c)) in enumerate(zip(minibatches_nc, minibatches_c)):
            # data
            inputs_c, targets_c = inputs_c.to(device), targets_c.to(device)
            inputs_nc, targets_nc = inputs_nc.to(device), targets_nc.to(device)
            
            # create a class-data dict for invariance loss
            class_ids_nc = {i: (targets_nc == i).nonzero().view(-1,1).type(torch.long) for i in range(n_classes)} 
            class_ids_c = {i: (targets_c == i).nonzero().view(-1,1).type(torch.long) for i in range(n_classes)}
            
            # ... Step 1: execute Barrier loss! ...
            # get linear interpolated model
            with torch.no_grad():
                t = 0.5 + 0.5 * (torch.randn(1)).clip(-0.5, 0.5)[0] 

                for module_c, module_nc, module_interpolated in zip(net_c.modules(), self.network.modules(), self.interpolated_network.modules()):
                    if isinstance(module_c, nn.Conv2d) or isinstance(module_c, nn.Linear):
                        module_interpolated.weight.data = t * module_c.weight.data + (1-t) * module_nc.weight.data
                        module_interpolated.bias.data = t * module_c.bias.data + (1-t) * module_nc.bias.data
                    elif isinstance(module_c, nn.BatchNorm2d):
                        module_interpolated.weight.data = t * module_c.weight.data + (1-t) * module_nc.weight.data
                        module_interpolated.bias.data = t * module_c.bias.data + (1-t) * module_nc.bias.data
            
            # ... compute loss and backward ...
            self.interpolated_network.zero_grad()
            outputs_interpolated = self.interpolated_network(inputs_c)
            loss = (self.loss_margin - self.criterion(outputs_interpolated, targets_c)).abs()
            loss.backward()

            # associate grads from interpolated model to theta_nc
            with torch.no_grad():
                for module_nc, module_interpolated in zip(self.network.modules(), self.interpolated_network.modules()):
                    if isinstance(module_nc, nn.Conv2d) or isinstance(module_nc, nn.Linear):
                        module_nc.weight.grad = (1-t) * module_interpolated.weight.grad.data
                        module_nc.bias.grad = (1-t) * module_interpolated.bias.grad.data
                    elif isinstance(module_nc, nn.BatchNorm2d):
                        module_nc.weight.grad = (1-t) * module_interpolated.weight.grad.data
                        module_nc.bias.grad = (1-t) * module_interpolated.bias.grad.data
            
            # ... Step 1 update ...
            self.optimizer.step()

            train_loss_c += loss.item()
            _, predicted_c = outputs_interpolated.max(1)
            total_c += targets_c.size(0)
            correct_c += predicted_c.eq(targets_c).sum().item()
            
            # ... Step 2: No-cue + Invariance loss ...
            self.network.zero_grad()
            if(epoch < self.warmup_epochs):
                outputs_ns = self.network(inputs_ns)
                loss = self.criterion(outputs_ns, targets_ns)
            else:
                z_nc, z_c = self.network(inputs_nc, use_linear=False), self.network(inputs_c, use_linear=False)
                inv_loss = 0
                for i in range(n_classes):
                    if (class_ids_nc[i].shape[0] == 0 or class_ids_c[i].shape[0] == 0):
                        continue
                    # MSE
                    inv_loss += (F.normalize(z_nc[class_ids_nc[i][:,0]].mean(dim=0, keepdim=True), dim=1) - F.normalize(z_c[class_ids_c[i][:,0]].mean(dim=0, keepdim=True), dim=1)).norm().pow(2)

                outputs_nc = self.network.linear(z_nc)
                loss = self.criterion(outputs_nc, targets_nc) + self.lambd * inv_loss / n_classes

            # ... Step 2 update ...
            loss.backward()
            self.optimizer.step()

            if epoch > self.warmup_epochs:
                self.lr_scheduler.step()

            train_loss_nc += loss.item()
            _, predicted_nc = outputs_nc.max(1)
            total_nc += targets_nc.size(0)
            correct_nc += predicted_nc.eq(targets_nc).sum().item()

            torch.cuda.empty_cache()

        return {'loss_c': train_loss_c/(batch_idx+1), 'accuracy_c': 100. * correct_c/total_c, 'loss_nc': train_loss_nc/(batch_idx+1), 'accuracy_nc': 100. * correct_nc/total_nc}


    # def predict(self, x):
    #     self.network.eval()
    #     with torch.no_grad():
    #         return self.network(x)


class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
            num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        input_grad = autograd.grad(
            F.cross_entropy(disc_out, disc_labels, reduction='sum'),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}




class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_meta_test = hparams['n_meta_test']

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in split_meta_train_test(minibatches, self.num_meta_test):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)


class CORAL_WL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL_WL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = 2

        all_x = torch.cat([x for x, y, u in minibatches])
        all_y = torch.cat([y for x, y, u in minibatches])
        all_u = torch.cat([u for x, y, u in minibatches])

        new_minibatches = []
        for i in range(nmb):
            idx = (all_u == i)
            x = all_x[idx]
            y = all_y[idx]
            new_minibatches.append((x,y))

        features = [self.featurizer(xi) for xi, _ in new_minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in new_minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class UShift(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(UShift, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)

        self.n_u = 2
        self.w = torch.zeros((self.n_u, 1))
        self.C = torch.zeros((self.n_u,self.n_u))
        self.num_classes = num_classes

        self.featurizer = networks.Featurizer(input_shape, self.hparams)

        self.classifier = nn.Sequential(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes * self.n_u,
                self.hparams['nonlinear_classifier']), 
            nn.Softmax(-2)
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)


        classifier_u = networks.Classifier(
            self.featurizer.n_outputs,
            self.n_u,
            self.hparams['nonlinear_classifier'])

        self.classifier_u = nn.Sequential(classifier_u, nn.Softmax(-1))

        # # Create a list of all parameters from featurizer, networks, and network_u
        # all_params = list(self.featurizer.parameters())
        # all_params += list(self.classifier.parameters())
        # all_params += list(self.classifier_u.parameters())

        classifier_lr_mult = 1 #self.hparams['mmd_gamma']

        # Define a single optimizer for all parameters
        self.optimizer = torch.optim.Adam(
            params = [
            {'params': self.featurizer.parameters()},
            {'params': self.classifier.parameters(), 'lr': self.hparams["lr"] * classifier_lr_mult},
            {'params': self.classifier_u.parameters(), 'lr': self.hparams["lr"]}
            ],
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = 2

        all_x = torch.cat([x for x, y, u in minibatches])
        all_y = torch.cat([y for x, y, u in minibatches])
        all_u = torch.cat([u for x, y, u in minibatches])

        # Find the minimum number of samples for any combination of y and u
        min_count = float('inf')
        for y_val in torch.unique(all_y):
            for u_val in torch.unique(all_u):
                count = ((all_y == y_val) & (all_u == u_val)).sum().item()
                if count < min_count:
                    min_count = count
        
        # Subsample all_x and all_y
        indices = []
        for y_val in torch.unique(all_y):
            for u_val in torch.unique(all_u):
                y_u_indices = ((all_y == y_val) & (all_u == u_val)).nonzero(as_tuple=True)[0]
                if len(y_u_indices) > 0 :
                    subsample_indices = torch.randperm(len(y_u_indices))[:min_count]
                    indices.append(y_u_indices[subsample_indices])
        
        indices = torch.cat(indices)
        all_x = all_x[indices]
        all_y = all_y[indices]
        all_u = all_u[indices]

        new_minibatches = []
        for i in range(nmb):
            idx = (all_u == i)
            x = all_x[idx]
            y = all_y[idx]
            u = all_u[idx]
            new_minibatches.append((x,y,u))

        features = [self.featurizer(xi) for xi, _, _ in new_minibatches]
        targets = [yi for _, yi,_ in new_minibatches]
        u_targets = [ui for _, _, ui in new_minibatches]
        classifs = [self.classifier(fi).reshape(-1, self.num_classes, self.n_u) for fi in features]
        classifs = [torch.bmm(y, F.one_hot(u, num_classes=self.n_u).float().unsqueeze(-1)).squeeze() for y,u in zip(classifs,targets)]
        u_classifs = [self.classifier_u(fi) for fi in features]

        objectives = []
        for i in range(nmb):
            obj_i = F.cross_entropy(classifs[i], targets[i])
            obj_i += F.cross_entropy(u_classifs[i], u_targets[i])
            objectives.append(obj_i)
            
            # Calculate and print accuracy for group i
            _, predicted = torch.max(classifs[i], 1)
            correct = (predicted == targets[i]).sum().item()
            total = targets[i].size(0)
            accuracy = 100 * correct / total

            # Calculate label proportions
            label_counts = torch.bincount(targets[i], minlength=self.num_classes)
            label_proportions = label_counts.float() / total

            # Calculate prediction proportions
            prediction_counts = torch.bincount(predicted, minlength=self.num_classes)
            prediction_proportions = prediction_counts.float() / total

            # Print statistics
            # print(f"group {i}: accuracy = {accuracy:.2f}%")
            # print(f"  Label proportions: {', '.join([f'{label}: {prop:.2f}' for label, prop in enumerate(label_proportions)])}")
            # print(f"  Prediction proportions: {', '.join([f'{label}: {prop:.2f}' for label, prop in enumerate(prediction_proportions)])}")

            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        # # Calculate weights based on the magnitude of each objective
        # objective_magnitudes = [obj.item() for obj in objectives]
        # total_magnitude = sum(objective_magnitudes)
        
        # # Avoid division by zero in case all magnitudes are zero
        # if total_magnitude == 0:
        #     weights = [1.0 / nmb] * nmb
        # else:
        #     weights = [mag / total_magnitude for mag in objective_magnitudes]

        # # Calculate weighted objective
        # weighted_objective = sum(w * o for w, o in zip(weights, objectives))
        weighted_objective = sum(0.5 * o for o in objectives)

        self.optimizer.zero_grad()
        (weighted_objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': weighted_objective.item(), 'penalty': penalty}
    
    def create_w(self, adapt_x):
        device = adapt_x.device
        mu = torch.zeros(self.n_u,1).to(device)
        matrices = []

        # features = self.featurizer(adapt_x)
        # u = self.classifieru(features)
        u = self.classifier_u(adapt_x)
        _, u = torch.max(u.unsqueeze(-1),1)
        for i in range(self.n_u):
            mu[i,0] = (u==i).sum().item()
        # print(mu)
        mu = mu/mu.sum()
        # print(mu)

        w = torch.linalg.pinv(self.C.to(adapt_x.device)) @ mu
        w = torch.clip(w, min=1e-2, max=15)
        w = w / w.sum()
        self.w = w.to(adapt_x.device)
    
    def create_C(self, all_x, all_u):
        device = all_x.device
        # features = self.featurizer(all_x)
        # u = self.classifiery(features)
        u = self.classifier_u(all_x)
        _, u = torch.max(u.unsqueeze(-1),1)
        cm = confusion_matrix(all_u.cpu(), u.cpu(), labels=list(range(self.n_u)), normalize='true')
        new_C = torch.from_numpy(np.array(cm)).float().T.to(device)
        

        self.C = new_C

    def predict(self, x, u=None):
        x = self.featurizer(x)

        # y = self.classifiery(features).view(-1,self.num_classes,self.n_u)
        # y = torch.cat([net(x).unsqueeze(-1) for net in self.networks],-1)
        
        
        if u is not None:
            y = self.classifier(x).reshape(-1,self.num_classes,self.n_u)
            all_u_one_hot = F.one_hot(u, num_classes=self.n_u).float().to(x.device)
            outputs = torch.bmm(y, all_u_one_hot.unsqueeze(-1)).squeeze()

            return outputs
        else:
            self.create_w(x)
            y = self.classifier(x).reshape(-1,self.num_classes,self.n_u)
            u = self.classifier_u(x).unsqueeze(-1)
            outputs = torch.bmm(y, u*self.w[None,...]).squeeze()
            
            return torch.log(outputs)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))

class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SD, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p ** 2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}

class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]

    def update(self, minibatches, unlabeled=None):
        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)

            env_grads = autograd.grad(env_loss, self.network.parameters())
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        self.optimizer.step()

        return {'loss': mean_loss}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0

class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, in_features, num_classes, num_domains, hparams):
        super(IGA, self).__init__(in_features, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=None):
        total_loss = 0
        grads = []
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grad = autograd.grad(env_loss, self.network.parameters(),
                                        create_graph=True)

            grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)
        mean_grad = autograd.grad(mean_loss, self.network.parameters(),
                                        retain_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.hparams['penalty'] * penalty_value

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': mean_loss.item(), 'penalty': penalty_value.item()}


class SelfReg(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SelfReg, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_classes = num_classes
        self.MSEloss = nn.MSELoss()
        input_feat_size = self.featurizer.n_outputs
        hidden_size = input_feat_size if input_feat_size==2048 else input_feat_size*2

        self.cdpl = nn.Sequential(
                            nn.Linear(input_feat_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, input_feat_size),
                            nn.BatchNorm1d(input_feat_size)
        )

    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        lam = np.random.beta(0.5, 0.5)

        batch_size = all_y.size()[0]

        # cluster and order features into same-class group
        with torch.no_grad():
            sorted_y, indices = torch.sort(all_y)
            sorted_x = torch.zeros_like(all_x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = all_x[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_y):
                if ex==val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            all_x = sorted_x
            all_y = sorted_y

        feat = self.featurizer(all_x)
        proj = self.cdpl(feat)

        output = self.classifier(feat)

        # shuffle
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0
        for end in intervals:
            shuffle_indices = torch.randperm(end-ex)+ex
            shuffle_indices2 = torch.randperm(end-ex)+ex
            for idx in range(end-ex):
                output_2[idx+ex] = output[shuffle_indices[idx]]
                feat_2[idx+ex] = proj[shuffle_indices[idx]]
                output_3[idx+ex] = output[shuffle_indices2[idx]]
                feat_3[idx+ex] = proj[shuffle_indices2[idx]]
            ex = end

        # mixup
        output_3 = lam*output_2 + (1-lam)*output_3
        feat_3 = lam*feat_2 + (1-lam)*feat_3

        # regularization
        L_ind_logit = self.MSEloss(output, output_2)
        L_hdl_logit = self.MSEloss(output, output_3)
        L_ind_feat = 0.3 * self.MSEloss(feat, feat_2)
        L_hdl_feat = 0.3 * self.MSEloss(feat, feat_3)

        cl_loss = F.cross_entropy(output, all_y)
        C_scale = min(cl_loss.item(), 1.)
        loss = cl_loss + C_scale*(lam*(L_ind_logit + L_ind_feat)+(1-lam)*(L_hdl_logit + L_hdl_feat))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SANDMask(ERM):
    """
    SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances in Domain Generalization
    <https://arxiv.org/abs/2106.02266>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]
        self.k = hparams["k"]
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            betas=betas
        )

        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):

        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)
            env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        # gradient masking applied here
        self.mask_grads(param_gradients, self.network.parameters())
        self.optimizer.step()
        self.update_count += 1

        return {'loss': mean_loss}

    def mask_grads(self, gradients, params):
        '''
        Here a mask with continuous values in the range [0,1] is formed to control the amount of update for each
        parameter based on the agreement of gradients coming from different environments.
        '''
        device = gradients[0][0].device
        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            avg_grad = torch.mean(grads, dim=0)
            grad_signs = torch.sign(grads)
            gamma = torch.tensor(1.0).to(device)
            grads_var = grads.var(dim=0)
            grads_var[torch.isnan(grads_var)] = 1e-17
            lam = (gamma * grads_var).pow(-1)
            mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
            mask = torch.max(mask, torch.zeros_like(mask))
            mask[torch.isnan(mask)] = 1e-17
            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))



class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)

class TRM(Algorithm):
    """
    Learning Representations that Support Robust Transfer of Predictors
    <https://arxiv.org/abs/2110.09940>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(TRM, self).__init__(input_shape, num_classes, num_domains,hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.num_domains = num_domains
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes).cuda()
        self.clist = [nn.Linear(self.featurizer.n_outputs, num_classes).cuda() for i in range(num_domains+1)]
        self.olist = [torch.optim.SGD(
            self.clist[i].parameters(),
            lr=1e-1,
        ) for i in range(num_domains+1)]

        self.optimizer_f = torch.optim.Adam(
            self.featurizer.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_c = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # initial weights
        self.alpha = torch.ones((num_domains, num_domains)).cuda() - torch.eye(num_domains).cuda()

    @staticmethod
    def neum(v, model, batch):
        def hvp(y, w, v):

            # First backprop
            first_grads = autograd.grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)
            first_grads = torch.nn.utils.parameters_to_vector(first_grads)
            # Elementwise products
            elemwise_products = first_grads @ v
            # Second backprop
            return_grads = autograd.grad(elemwise_products, w, create_graph=True)
            return_grads = torch.nn.utils.parameters_to_vector(return_grads)
            return return_grads

        v = v.detach()
        h_estimate = v
        cnt = 0.
        model.eval()
        iter = 10
        for i in range(iter):
            model.weight.grad *= 0
            y = model(batch[0].detach())
            loss = F.cross_entropy(y, batch[1].detach())
            hv = hvp(loss, model.weight, v)
            v -= hv
            v = v.detach()
            h_estimate = v + h_estimate
            h_estimate = h_estimate.detach()
            # not converge
            if torch.max(abs(h_estimate)) > 10:
                break
            cnt += 1

        model.train()
        return h_estimate.detach()

    def update(self, minibatches, unlabeled=None):

        loss_swap = 0.0
        trm = 0.0

        if self.update_count >= self.hparams['iters']:
            # TRM
            if self.hparams['class_balanced']:
                # for stability when facing unbalanced labels across environments
                for classifier in self.clist:
                    classifier.weight.data = copy.deepcopy(self.classifier.weight.data)
            self.alpha /= self.alpha.sum(1, keepdim=True)

            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            # updating original network
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

            for i in range(30):
                all_logits_idx = 0
                loss_erm = 0.
                for j, (x, y) in enumerate(minibatches):
                    # j-th domain
                    feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                    all_logits_idx += x.shape[0]
                    loss_erm += F.cross_entropy(self.clist[j](feature.detach()), y)
                for opt in self.olist:
                    opt.zero_grad()
                loss_erm.backward()
                for opt in self.olist:
                    opt.step()

            # collect (feature, y)
            feature_split = list()
            y_split = list()
            all_logits_idx = 0
            for i, (x, y) in enumerate(minibatches):
                feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                feature_split.append(feature)
                y_split.append(y)

            # estimate transfer risk
            for Q, (x, y) in enumerate(minibatches):
                sample_list = list(range(len(minibatches)))
                sample_list.remove(Q)

                loss_Q = F.cross_entropy(self.clist[Q](feature_split[Q]), y_split[Q])
                grad_Q = autograd.grad(loss_Q, self.clist[Q].weight, create_graph=True)
                vec_grad_Q = nn.utils.parameters_to_vector(grad_Q)

                loss_P = [F.cross_entropy(self.clist[Q](feature_split[i]), y_split[i])*(self.alpha[Q, i].data.detach())
                          if i in sample_list else 0. for i in range(len(minibatches))]
                loss_P_sum = sum(loss_P)
                grad_P = autograd.grad(loss_P_sum, self.clist[Q].weight, create_graph=True)
                vec_grad_P = nn.utils.parameters_to_vector(grad_P).detach()
                vec_grad_P = self.neum(vec_grad_P, self.clist[Q], (feature_split[Q], y_split[Q]))

                loss_swap += loss_P_sum - self.hparams['cos_lambda'] * (vec_grad_P.detach() @ vec_grad_Q)

                for i in sample_list:
                    self.alpha[Q, i] *= (self.hparams["groupdro_eta"] * loss_P[i].data).exp()

            loss_swap /= len(minibatches)
            trm /= len(minibatches)
        else:
            # ERM
            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

        nll = loss.item()
        self.optimizer_c.zero_grad()
        self.optimizer_f.zero_grad()
        if self.update_count >= self.hparams['iters']:
            loss_swap = (loss + loss_swap)
        else:
            loss_swap = loss

        loss_swap.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        loss_swap = loss_swap.item() - nll
        self.update_count += 1

        return {'nll': nll, 'trm_loss': loss_swap}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def train(self):
        self.featurizer.train()

    def eval(self):
        self.featurizer.eval()

class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IB_penalty': ib_penalty.item()}

class IB_IRM(ERM):
    """Information Bottleneck based IRM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        irm_penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        irm_penalty = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            irm_penalty += self._irm_penalty(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        irm_penalty /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += irm_penalty_weight * irm_penalty
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['irm_penalty_anneal_iters'] or self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IRM_penalty': irm_penalty.item(),
                'IB_penalty': ib_penalty.item()}


class AbstractCAD(Algorithm):
    """Contrastive adversarial domain bottleneck (abstract class)
    from Optimal Representations for Covariate Shift <https://arxiv.org/abs/2201.00057>
    """

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, is_conditional):
        super(AbstractCAD, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        params = list(self.featurizer.parameters()) + list(self.classifier.parameters())

        # parameters for domain bottleneck loss
        self.is_conditional = is_conditional  # whether to use bottleneck conditioned on the label
        self.base_temperature = 0.07
        self.temperature = hparams['temperature']
        self.is_project = hparams['is_project']  # whether apply projection head
        self.is_normalized = hparams['is_normalized'] # whether apply normalization to representation when computing loss

        # whether flip maximize log(p) (False) to minimize -log(1-p) (True) for the bottleneck loss
        # the two versions have the same optima, but we find the latter is more stable
        self.is_flipped = hparams["is_flipped"]

        if self.is_project:
            self.project = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, 128),
            )
            params += list(self.project.parameters())

        # Optimizers
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def bn_loss(self, z, y, dom_labels):
        """Contrastive based domain bottleneck loss
         The implementation is based on the supervised contrastive loss (SupCon) introduced by
         P. Khosla, et al., in “Supervised Contrastive Learning“.
        Modified from  https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
        """
        device = z.device
        batch_size = z.shape[0]

        y = y.contiguous().view(-1, 1)
        dom_labels = dom_labels.contiguous().view(-1, 1)
        mask_y = torch.eq(y, y.T).to(device)
        mask_d = (torch.eq(dom_labels, dom_labels.T)).to(device)
        mask_drop = ~torch.eye(batch_size).bool().to(device)  # drop the "current"/"self" example
        mask_y &= mask_drop
        mask_y_n_d = mask_y & (~mask_d)  # contain the same label but from different domains
        mask_y_d = mask_y & mask_d  # contain the same label and the same domain
        mask_y, mask_drop, mask_y_n_d, mask_y_d = mask_y.float(), mask_drop.float(), mask_y_n_d.float(), mask_y_d.float()

        # compute logits
        if self.is_project:
            z = self.project(z)
        if self.is_normalized:
            z = F.normalize(z, dim=1)
        outer = z @ z.T
        logits = outer / self.temperature
        logits = logits * mask_drop
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        if not self.is_conditional:
            # unconditional CAD loss
            denominator = torch.logsumexp(logits + mask_drop.log(), dim=1, keepdim=True)
            log_prob = logits - denominator

            mask_valid = (mask_y.sum(1) > 0)
            log_prob = log_prob[mask_valid]
            mask_d = mask_d[mask_valid]

            if self.is_flipped:  # maximize log prob of samples from different domains
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (~mask_d).float().log(), dim=1)
            else:  # minimize log prob of samples from same domain
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (mask_d).float().log(), dim=1)
        else:
            # conditional CAD loss
            if self.is_flipped:
                mask_valid = (mask_y_n_d.sum(1) > 0)
            else:
                mask_valid = (mask_y_d.sum(1) > 0)

            mask_y = mask_y[mask_valid]
            mask_y_d = mask_y_d[mask_valid]
            mask_y_n_d = mask_y_n_d[mask_valid]
            logits = logits[mask_valid]

            # compute log_prob_y with the same label
            denominator = torch.logsumexp(logits + mask_y.log(), dim=1, keepdim=True)
            log_prob_y = logits - denominator

            if self.is_flipped:  # maximize log prob of samples from different domains and with same label
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_n_d.log(), dim=1)
            else:  # minimize log prob of samples from same domains and with same label
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_d.log(), dim=1)

        def finite_mean(x):
            # only 1D for now
            num_finite = (torch.isfinite(x).float()).sum()
            mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
            if num_finite != 0:
                mean = mean / num_finite
            else:
                return torch.tensor(0.0).to(x)
            return mean

        return finite_mean(bn_loss)

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        all_d = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        bn_loss = self.bn_loss(all_z, all_y, all_d)
        clf_out = self.classifier(all_z)
        clf_loss = F.cross_entropy(clf_out, all_y)
        total_loss = clf_loss + self.hparams['lmbda'] * bn_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"clf_loss": clf_loss.item(), "bn_loss": bn_loss.item(), "total_loss": total_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class CAD(AbstractCAD):
    """Contrastive Adversarial Domain (CAD) bottleneck

       Properties:
       - Minimize I(D;Z)
       - Require access to domain labels but not task labels
       """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=False)


class CondCAD(AbstractCAD):
    """Conditional Contrastive Adversarial Domain (CAD) bottleneck

    Properties:
    - Minimize I(D;Z|Y)
    - Require access to both domain labels and task labels
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CondCAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=True)


class Transfer(Algorithm):
    '''Algorithm 1 in Quantifying and Improving Transferability in Domain Generalization (https://arxiv.org/abs/2106.03632)'''
    ''' tries to ensure transferability among source domains, and thus transferabiilty between source and target'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Transfer, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.d_steps_per_g = hparams['d_steps_per_g']

        # Architecture
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.adv_classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.adv_classifier.load_state_dict(self.classifier.state_dict())

        # Optimizers
        if self.hparams['gda']:
            self.optimizer = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr'])
        else:
            self.optimizer = torch.optim.Adam(
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.adv_opt = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr_d'])

    def loss_gap(self, minibatches, device):
        ''' compute gap = max_i loss_i(h) - min_j loss_j(h), return i, j, and the gap for a single batch'''
        max_env_loss, min_env_loss =  torch.tensor([-float('inf')], device=device), torch.tensor([float('inf')], device=device)
        for x, y in minibatches:
            p = self.adv_classifier(self.featurizer(x))
            loss = F.cross_entropy(p, y)
            if loss > max_env_loss:
                max_env_loss = loss
            if loss < min_env_loss:
                min_env_loss = loss
        return max_env_loss - min_env_loss

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        # outer loop
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del all_x, all_y
        gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
        self.optimizer.zero_grad()
        gap.backward()
        self.optimizer.step()
        self.adv_classifier.load_state_dict(self.classifier.state_dict())
        for _ in range(self.d_steps_per_g):
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
        return {'loss': loss.item(), 'gap': -gap.item()}

    def update_second(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count = (self.update_count + 1) % (1 + self.d_steps_per_g)
        if self.update_count.item() == 1:
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            loss = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del all_x, all_y
            gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            self.optimizer.zero_grad()
            gap.backward()
            self.optimizer.step()
            self.adv_classifier.load_state_dict(self.classifier.state_dict())
            return {'loss': loss.item(), 'gap': gap.item()}
        else:
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
            return {'gap': -gap.item()}


    def predict(self, x):
        return self.classifier(self.featurizer(x))


class AbstractCausIRL(ERM):
    '''Abstract class for Causality based invariant representation learning algorithm from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractCausIRL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        first = None
        second = None

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i] + 1e-16, targets[i])
            slice = np.random.randint(0, len(features[i]))
            if first is None:
                first = features[i][:slice]
                second = features[i][slice:]
            else:
                first = torch.cat((first, features[i][:slice]), 0)
                second = torch.cat((second, features[i][slice:]), 0)
        if len(first) > 1 and len(second) > 1:
            penalty = torch.nan_to_num(self.mmd(first, second))
        else:
            penalty = torch.tensor(0)
        objective /= nmb

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class CausIRL_MMD(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the MMD distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_MMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, gaussian=True)


class CausIRL_CORAL(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the CORAL distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_CORAL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, gaussian=False)
