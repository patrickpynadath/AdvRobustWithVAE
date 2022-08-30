from Training.train_adversarial import AdvTrainer
from Tests.classifier_test import ClassifierTest
from Experiments.exp import Experiment

# the goal of this experiment is to get data for adversarial training on
# - a base classifier
# - the Smooth counterpart, based on work from Cohen et al and Salman et al
# - the SmoothVAE counterpart
# this uses a convolutional neural network classifier as the base
# and a convolutional VAE

class AdvExperiment(Experiment):
    def __init__(self, batch_size, log_dir, out_dir, device):
        super().__init__( batch_size, log_dir, out_dir, device)
        self.out_dir += 'advtrain'

    def adv_exp_base(self, train_eps_value, train_steps, lr, train_adversary_type, clf_epochs
                     , test_num_attacks, test_attack_eps, test_adversary_type, test_attack_steps):
        label = f'base_clf_advtrain_{round(train_eps_value, 4)}'
        models, optimizers, criterions = self.get_base_models(label, lr)
        adv_trainer = AdvTrainer(models, self.trainloader, self.testloader, self.device, optimizers, criterions,
                                 train_eps_value, train_steps, train_adversary_type, self.log_dir, use_tensorboard=True)
        adv_trainer.training_loop(clf_epochs)
        adv_tester = ClassifierTest(models, self.testloader, self.device, self.batch_size, self.out_dir)
        adv_tester.test_clean()
        adv_tester.test_adv(test_adversary_type, test_attack_eps, test_attack_steps, test_num_attacks)
        adv_tester.to_file()
        return

    def adv_exp_smooth(self, sigmas, train_eps_value, train_steps, lr, train_adversary_type, clf_epochs
                     , m_train, test_num_attacks, test_attack_eps, test_adversary_type, test_attack_steps):
        base_label = f'advtrain{round(train_eps_value, 4)}_'
        models, optimizers, criterions = self.get_smooth_models(base_label, lr, sigmas, m_train)
        adv_trainer = AdvTrainer(models, self.trainloader, self.testloader, self.device, optimizers,
                                 criterions, train_eps_value, train_steps, train_adversary_type, self.log_dir, use_tensorboard=True)
        adv_trainer.training_loop(clf_epochs)
        adv_tester = ClassifierTest(models, self.testloader, self.device, self.batch_size, self.out_dir)
        adv_tester.test_clean()
        adv_tester.test_adv(test_adversary_type, test_attack_eps, test_attack_steps, test_num_attacks)
        adv_tester.to_file()
        return

    def adv_exp_smoothVAE(self, sigmas, train_eps_value, train_steps, lr, train_adversary_type, clf_epochs, vae_epochs
                     , m_train, test_num_attacks, test_attack_eps, test_adversary_type, test_attack_steps, with_VAE_grad = False):
        base_label = f'advtrain{round(train_eps_value, 4)}_'
        models, optimizers, criterions = self.get_smoothVAE_models(base_label, lr, sigmas, m_train, vae_epochs, with_VAE_grad)
        adv_trainer = AdvTrainer(models, self.trainloader, self.testloader, self.device, optimizers,
                                 criterions, train_eps_value, train_steps, train_adversary_type, self.log_dir,
                                 use_tensorboard=True)
        adv_trainer.training_loop(clf_epochs)
        adv_tester = ClassifierTest(models, self.testloader, self.device, self.batch_size, self.out_dir)
        adv_tester.test_clean()
        adv_tester.test_adv(test_adversary_type, test_attack_eps, test_attack_steps, test_num_attacks)
        adv_tester.to_file()
        return
