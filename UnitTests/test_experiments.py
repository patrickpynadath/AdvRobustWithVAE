import unittest
import torch
from Experiments import BaseExp
import os

os.chdir("../")
class TestBaseExp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base_exp = BaseExp(training_logdir='../TestDir/TrainingDir/',
                                exp_logdir='../TestDir/ExperimentDir/',
                                device='cpu')
        cls.base_exp = base_exp
        cls.resnet = base_exp.get_trained_resnet(net_depth=20,
                                                  block_name='BasicBlock',
                                                  batch_size=100,
                                                  optimizer='adam',
                                                  lr=.01,
                                                  epochs=1,
                                                 use_step_lr=False,
                                                 lr_schedule_step=1,
                                                 lr_schedule_gamma=1)

    def test_initialization(self):
        if not self.base_exp.train_set:
            self.fail("base_exp.train_set not initialized")
        if not self.base_exp.test_set:
            self.fail("base_exp.test_set not initialized")

    def test_get_loaders(self):
        train_loader, test_loader = self.base_exp.get_loaders(batch_size=100)

        it = iter(train_loader)
        data, labels = next(it)
        self.assertEqual(data.size(), (100, 3, 32, 32), "batch data dimensions incorrect for train loader")
        self.assertEqual(labels.size(), (100,), "batch label dimensions incorrect for train loader")

        it = iter(test_loader)
        data, labels = next(it)
        self.assertEqual(data.size(), (100, 3, 32, 32), "batch data dimensions incorrect for test loader")
        self.assertEqual(labels.size(), (100,), "batch label dimensions incorrect for test loader")

    def test_get_adv_ex(self):
        resnet = self.resnet
        for adv_type in ['l2', 'linf']:
            test_adv_eps = .1
            original_samples, adv_samples, labels = self.base_exp.get_adv_examples(trained_clf=resnet,
                                                                                   attack_eps=test_adv_eps,
                                                                                   adversary_type=adv_type,
                                                                                   steps=1,
                                                                                   num_attacks=2,
                                                                                   dataset_name='train')
            self.assertEqual(original_samples.size(), (2, 3, 32, 32), "original images dim do not match")
            self.assertEqual(adv_samples.size(), (2, 3, 32, 32), "adv images dim do not match")
            self.assertEqual(labels.size(), (2,), "labels dim do not match")
            self.assertEqual(torch.equal(original_samples, adv_samples), False, "adv samples should not be same as original")
            for i in range(2):
                dist = torch.linalg.vector_norm(torch.flatten(original_samples[i, :] - adv_samples[i, :]), ord=float('inf'))
                self.assertLessEqual(dist, test_adv_eps)

    def test_eval_clf_clean(self):
        resnet = self.resnet
        nat_acc = self.base_exp.eval_clf_clean(resnet)
        self.assertGreaterEqual(nat_acc, 0, "Accuracy should not be less than 0")
        self.assertLessEqual(nat_acc, 1, "Accuracy should not be greater than 1")


if __name__ == '__main__':
    unittest.main()
