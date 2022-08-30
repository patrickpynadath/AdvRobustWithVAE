import torch
import numpy as np
import torchattacks
from Adversarial.attacks import PGD_L2
import torch.nn as f



# Class for running tests against a given model -- both natural accuracy and adversarial robustness
class ClassifierTest:
    def __init__(self,
                 model : f.Module,
                 testloader,
                 device,
                 batch_size):
        """

        :param model: f.Module to be evaluated using testloader data
        :param testloader: torch DataLoader to get test samples from
        :param device: device to put model and data on
        :param batch_size:
        """
        self.model = model
        self.testloader = testloader
        self.device = device
        self.batch_size = batch_size


    def test_clean(self):
        """

        :return: the natural accuracy of self.model on samples from self.testloader
        """
        num_correct = 0
        model = self.model
        model.eval()
        total = 0
        for i, data in enumerate(self.testloader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            raw_res = model(inputs)
            base_pred = torch.argmax(raw_res, dim=1)
            num_correct += sum([1 if base_pred[i].item() == labels[i].item() else 0 for i in range(len(base_pred))])
            total += len(inputs)
        nat_acc = num_correct / total
        return nat_acc

    def test_adv(self,
                 adversary_type,
                 attack_eps_values,
                 steps,
                 num_attacks):
        """

        :param adversary_type: type of attack to use against model, either 'linf' or 'l2'
        :param attack_eps_values: max norm values for attacker
        :param steps: number of steps for PGD attack
        :param num_attacks: how many samples to use attack model with
        :return: accuracy, which is num_correct/num_attacks
        """
        testset = self.testloader.dataset
        samples_idx = np.random.randint(low=0, high=len(testset), size=num_attacks)
        original_im_dim = tuple([num_attacks]) + testset[0][0].size()
        original_im = torch.zeros(original_im_dim)
        labels = torch.zeros(size=(num_attacks,))
        for i, idx in enumerate(samples_idx):
            original_im[i, :] = testset[idx][0]
            labels[i] = testset[idx][1]
        labels = labels.type(torch.LongTensor)
        original_im, labels = original_im.to(self.device), labels.to(self.device)
        scores = [0 for _ in range(len(attack_eps_values))]
        model = self.model
        attacks = torch.zeros_like(original_im).to(self.device)
        for i, attack_eps in enumerate(attack_eps_values):
            if adversary_type == 'l2':
                attacker = PGD_L2(model, steps = steps, max_norm=attack_eps, device=self.device)
                attacks.add(attacker.attack(original_im, labels))
            elif adversary_type == 'linf':
                attacker = torchattacks.PGD(model, attack_eps, steps)
                attacks.add(attacker(original_im, labels).to(self.device))
            base_pred = torch.argmax(model(attacks), dim=1)
            scores[i] = sum(
                [1 if base_pred[i].item() == labels[i].item() else 0 for i in range(num_attacks)]) / num_attacks
            attacks.zero_()
        return scores
