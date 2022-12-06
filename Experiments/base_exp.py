import torch

from Utils import get_cifar_sets
from Experiments.helper_functions import get_adv_examples
from Models import Smooth, SmoothSoftClf
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseExp:
    def __init__(self,
                 device,
                 batch_size=128):

        self.device = device
        train_set, test_set = get_cifar_sets()
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.num_classes = len(classes)
        self.train_loader = train_loader
        self.batch_size = batch_size
        self.test_loader = test_loader
        self.device = device

    def eval_clf_clean(self, model):
        num_correct = 0
        model.eval()
        model.to(self.device)
        total = 0
        pg = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        pg.set_description("Natural Accuracy")
        for i, data in pg:
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            raw_res = model(inputs)
            base_pred = torch.argmax(raw_res, dim=1)
            num_correct += sum([1 if base_pred[i].item() == labels[i].item() else 0 for i in range(len(base_pred))])
            total += len(inputs)
        nat_acc = num_correct / total
        print(f"Nat Acc : {round(nat_acc * 100, 3)}%")
        return nat_acc

    def eval_clf_adv_raw(self,
                         model,
                         adversary_type,
                         eps_values,
                         steps):
        adv_accs = []
        model.to(self.device)
        model.eval()
        for eps in eps_values:
            num_correct = 0
            total = 0
            pg = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
            pg.set_description(f'Adv Acc for {adversary_type} @ eps {round(eps, 5)}')
            for i, data in pg:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                attacked_inputs = get_adv_examples(model, attack_eps=eps, adversary_type=adversary_type,
                                                   steps=steps, nat_img=inputs, labels=labels)
                raw_res = model(attacked_inputs)
                base_pred = torch.argmax(raw_res, dim=1)
                num_correct += sum([1 if base_pred[i].item() == labels[i].item() else 0 for i in range(len(base_pred))])
                total += len(inputs)
            raw_acc = num_correct / total
            print(f"Raw adv acc {round(raw_acc * 100, 3)}%")
            adv_accs.append(raw_acc)
        return adv_accs

    def eval_smoothclf_nat_raw(self, model, smoothSigma, num_samples, conf_value):
        num_correct = 0
        model.eval()
        model.to(self.device)
        smooth = Smooth(model, 10, smoothSigma)
        total = 0
        pg = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        pg.set_description("Natural Accuracy")
        for i, data in pg:
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            for i in range(len(labels)):
                pred = smooth.predict(inputs[i, :], num_samples, conf_value, batch_size=100)
                total += 1
                if pred == labels[i].item():
                    num_correct += 1
            del inputs
            del labels
        nat_acc = num_correct / total
        print(f"Nat Acc : {round(nat_acc * 100, 3)}%")
        return nat_acc



    def eval_smoothclf_adv_raw(self,
                               model,
                               smoothSigma,
                               num_samples,
                               adversary_type,
                               conf_value,
                               eps_values,
                               steps):
        adv_accs = []
        model.to(self.device)
        model.eval()
        smooth = Smooth(model, 10, smoothSigma)
        attack_model = SmoothSoftClf(model, smoothSigma, self.device)
        for eps in eps_values:
            num_correct = 0
            total = 0
            pg = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
            pg.set_description(f'Adv Acc for {adversary_type} @ eps {round(eps, 5)}')
            for i, data in pg:
                inputs, labels = data[0].to(self.device), data[1]
                attack_model.eval()
                print('asd')
                attacked_inputs = get_adv_examples(attack_model, attack_eps=eps, adversary_type=adversary_type,
                                                   steps=steps, nat_img=inputs, labels=labels)
                for i in range(len(labels)):
                    pred_class = smooth.predict(attacked_inputs[i, :], num_samples, conf_value, batch_size=10)
                    total += 1
                    if pred_class == labels[i].item():
                        num_correct += 1
                del inputs
                del labels
            raw_acc = num_correct / total
            print(f"Raw adv acc {round(raw_acc * 100, 3)}%")
            adv_accs.append(raw_acc)
        return adv_accs


