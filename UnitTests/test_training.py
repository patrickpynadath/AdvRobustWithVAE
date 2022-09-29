from unittest import TestCase
from torchtest import test_suite
import torch
from Models import ResNet, Conv_VAE, SmoothVAE_Sample, SmoothVAE_Latent

class Test_Training_ResNet(TestCase):

    def test_training_step(self):
        model = ResNet(depth=20, num_classes=10, block_name='BasicBlock')
        loss_fn = torch.nn.functional.cross_entropy
        optim = torch.optim.Adam(model.parameters(), lr=.01)
        inputs = torch.rand((10, 3, 32, 32))
        labels = torch.rand((10, 10))
        test_suite(model, loss_fn, optim, [inputs, labels],
                   train_vars=model.named_parameters(),
                   test_vars_change=True,
                   test_nan_vals=True,
                   test_inf_vals=True)


class Test_Training_VAE(TestCase):
    def test_training_step(self):
        model = Conv_VAE(32, 3, 16, 20, 'cpu')
        inputs = torch.rand((10, 3, 32, 32))
        labels = torch.rand((10, 10))
        optim = torch.optim.Adam(model.parameters(), lr=.01)
        test_suite(model, model.loss_fn, optim, [inputs, labels],
                   train_vars=model.named_parameters(),
                   test_vars_change=True,
                   test_inf_vals=True,
                   test_nan_vals=True)


class Test_Training_SmoothVAE(TestCase):

    def test_training_step_sample(self):
        clf = ResNet(depth=20, num_classes=10, block_name='BasicBlock')
        vae = Conv_VAE(32, 3, 16, 20, 'cpu')
        model = SmoothVAE_Sample(clf, 1, vae, 'cpu', num_samples=10, num_classes=10, loss_coef=1, vae_param=False)
        inputs = torch.rand((10, 3, 32, 32))
        labels = torch.rand((10, 10))
        loss_fn = torch.nn.functional.cross_entropy
        optim = torch.optim.Adam(model.parameters(), lr=.01)
        test_suite(model, loss_fn, optim, [inputs, labels],
                   non_train_vars=vae.named_parameters(),
                   train_vars=clf.named_parameters(),
                   test_nan_vals=True,
                   test_inf_vals=True)


        model = SmoothVAE_Sample(clf, 1, vae, 'cpu', num_samples=10, num_classes=10, loss_coef=1, vae_param=True)
        test_suite(model, loss_fn, optim, [inputs, labels],
                   train_vars=model.main_module.named_parameters(),
                   test_nan_vals=True,
                   test_inf_vals=True)



    def test_training_step_latent(self):
        clf = ResNet(depth=20, num_classes=10, block_name='BasicBlock')
        vae = Conv_VAE(32, 3, 16, 20, 'cpu')
        model = SmoothVAE_Latent(clf, 1, vae, 'cpu', num_samples=10, num_classes=10, loss_coef=1,
                                 vae_param=False)
        inputs = torch.rand((10, 3, 32, 32)).to('cpu')
        labels = torch.rand((10, 10)).to('cpu')
        loss_fn = torch.nn.functional.cross_entropy
        optim = torch.optim.Adam(model.parameters(), lr=.01)
        test_suite(model, loss_fn, optim, [inputs, labels],
                   train_vars=clf.named_parameters(),
                   non_train_vars=vae.named_parameters(),
                   test_vars_change=True,
                   test_nan_vals=True,
                   test_inf_vals=True)


        model = SmoothVAE_Latent(clf, 1, vae, 'cpu', num_samples=10, num_classes=10, loss_coef=1, vae_param=True)
        test_suite(model, loss_fn, optim, [inputs, labels],
                   train_vars=model.main_module.named_parameters(),
                   test_vars_change=True,
                   test_nan_vals=True,
                   test_inf_vals=True)




