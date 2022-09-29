import unittest
from Models import ResNet, Conv_VAE, SmoothVAE_Sample, SmoothVAE_Latent
import torch

class TestResNet(unittest.TestCase):


    # ensuring that if an incorrect depth is passed, an error is thrown
    def test_resnet_constructor(self):
        self.assertRaises(AssertionError, ResNet, depth=21)

    # testing the dimensions of outputs for each layer in resnet
    def test_resnet_basic(self):
        ex_resnet = ResNet(depth=20, block_name='BasicBlock', num_classes=10)
        # cifar is batch_size x 3 x 32 x 32
        x = torch.rand((10, 3, 32, 32))
        out = ex_resnet(x)
        self.assertEqual(out.size(), (10, 10),
                         msg="Output of resnet does not match expected dim of batch_size x num_classes")

    # def test_resnet_bottleneck(self):
    #     ex_resnet = ResNet(depth=20, block_name='BottleNeck', num_classes='10')
    #     x = torch.rand((10, 3, 32, 32))
    #     out = ex_resnet(x)
    #     self.assertEqual(out.size(), (10, 10),
    #                      msg = "Output of resnet does not match expected dim of batch_size x num_classes")


class TestConvVAE(unittest.TestCase):

    # testing whether the vae is outpoutting the expected dimensions at each step
    def test_conv_vae_dimensions(self):
        orig_dim = (10, 3, 32, 32)
        feature_size = 32 // 8
        x = torch.rand(orig_dim)
        kern_num = 16
        z_size = 20
        ex_vae = Conv_VAE(32, 3, kern_num, z_size, 'cpu', beta=1)
        encoded = ex_vae.encoder(x)
        self.assertEqual(encoded.size(), (10, kern_num, 4, 4))
        mean, logvar = ex_vae.q(encoded)
        z = ex_vae.z(mean, logvar)
        self.assertEqual(mean.size(), (10, z_size))
        self.assertEqual(mean.size(), (10, z_size))
        self.assertEqual(z.size(), (10, z_size))
        z_projected = ex_vae.project(z).view(
            -1, kern_num, feature_size, feature_size,)
        recon = ex_vae.decoder(z_projected)
        self.assertEqual(recon.size(), orig_dim)










