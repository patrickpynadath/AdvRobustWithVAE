from Experiments.base_exp import BaseExp
from torch.utils.tensorboard import SummaryWriter


class AdvRobustnessNaturalTraining(BaseExp):

    def adv_rob_base_clf(self,
                         net_depth,
                         clf_epochs,
                         adv_norms,
                         adv_type,
                         use_step_lr,
                         lr_schedule_step,
                         lr_schedule_gamma,
                         block_name="BottleNeck",
                         batch_size=256,
                         optimizer='sgd',
                         lr=.1,
                         adv_steps=10,
                         num_attacks=1000,
                         dataset_name='test'):
        resnet_clf = self.get_trained_resnet(net_depth=net_depth,
                                             block_name=block_name,
                                             batch_size=batch_size,
                                             optimizer=optimizer,
                                             lr=lr,
                                             epochs=clf_epochs,
                                             use_step_lr=use_step_lr,
                                             lr_schedule_step=lr_schedule_step,
                                             lr_schedule_gamma=lr_schedule_gamma)

        nat_acc, adv_accuracies = self.get_accuracies(clf=resnet_clf,
                                                      adv_norms=adv_norms,
                                                      adv_type=adv_type,
                                                      adv_steps=adv_steps,
                                                      num_attacks=num_attacks,
                                                      dataset_name=dataset_name)
        return nat_acc, adv_accuracies

    def adv_rob_smoothclf(self,
                          clf_epochs,
                          net_depth,
                          smoothing_sigma,
                          adv_norms,
                          adv_type,
                          use_step_lr,
                          lr_schedule_step,
                          lr_schedule_gamma,
                          clf_batch_size=100,
                          block_name="BasicBlock",
                          optimizer='adam',
                          lr=.01,
                          m_train=10,
                          adv_steps=10,
                          num_attacks=1000,
                          dataset_name='test'):
        """
        :param lr_schedule_gamma:
        :param use_step_lr:
        :param lr_schedule_step:
        :param dataset_name:
        :param net_depth:
        :param block_name:
        :param m_train:
        :param lr:
        :param optimizer:
        :param clf_batch_size:
        :param clf_epochs: epochs to train classifier with
        :param smoothing_sigma: smoothing value for randomized smoothing procedure
        :param adv_type: 'l2' or 'linf'
        :param adv_norms: list of max norms for PGD attack
        :param adv_steps: number of steps to use for PGD attack
        :param num_attacks: number of adversarial examples to evaluate trained model against
        :return: natural accuracy of model, list of adversarial robustness, and label of model
        """
        smooth_resnet = self.get_trained_smooth_resnet(net_depth=net_depth,
                                                       block_name=block_name,
                                                       m_train=m_train,
                                                       batch_size=clf_batch_size,
                                                       optimizer=optimizer,
                                                       lr=lr,
                                                       epochs=clf_epochs,
                                                       smoothing_sigma=smoothing_sigma,
                                                       use_step_lr=use_step_lr,
                                                       lr_schedule_step=lr_schedule_step,
                                                       lr_schedule_gamma=lr_schedule_gamma)
        nat_acc, adv_accuracies = self.get_accuracies(clf=smooth_resnet,
                                                      adv_norms=adv_norms,
                                                      adv_type=adv_type,
                                                      adv_steps=adv_steps,
                                                      num_attacks=num_attacks,
                                                      dataset_name=dataset_name)
        return nat_acc, adv_accuracies

    # need to get VAE as well, so need to also pass those hyperparam in
    def adv_rob_smoothvae_clf(self,
                              clf_epochs,
                              vae_epochs,
                              net_depth,
                              smoothing_vae_sigma,
                              adv_norms,
                              adv_type='linf',
                              use_step_lr=True,
                              lr_schedule_step=50,
                              lr_schedule_gamma = .1,
                              m_train=10,
                              smooth_vae_version='sample',
                              adv_steps=10,
                              num_attacks=1000,
                              block_name='BasicBlock',
                              dataset_name='test',
                              optimizer='sgd',
                              vae_batch_size=32,
                              clf_batch_size=100,
                              clf_lr=.01):
        resnet_smooth_vae = self.get_trained_smooth_vae_resnet(net_depth=net_depth,
                                                               block_name=block_name,
                                                               m_train=m_train,
                                                               batch_size_clf=clf_batch_size,
                                                               batch_size_vae=vae_batch_size,
                                                               optimizer=optimizer,
                                                               lr_clf=clf_lr,
                                                               epochs_clf=clf_epochs,
                                                               epochs_vae=vae_epochs,
                                                               smoothing_sigma=smoothing_vae_sigma,
                                                               smooth_vae_version=smooth_vae_version,
                                                               use_vae_param=False,
                                                               use_step_lr=use_step_lr,
                                                               lr_schedule_step=lr_schedule_step,
                                                               lr_schedule_gamma=lr_schedule_gamma)
        nat_acc, adv_accuracies = self.get_accuracies(clf=resnet_smooth_vae,
                                                      adv_norms=adv_norms,
                                                      adv_type=adv_type,
                                                      adv_steps=adv_steps,
                                                      num_attacks=num_attacks,
                                                      dataset_name=dataset_name)
        return nat_acc, adv_accuracies

    def get_accuracies(self,
                       clf,
                       adv_norms,
                       adv_type,
                       adv_steps,
                       num_attacks,
                       dataset_name):
        nat_acc = self.eval_clf_clean(model=clf)
        adv_accuracies = []
        for attack_eps in adv_norms:
            adv_accuracy = self.eval_clf_adv(model=clf,
                                             adversary_type=adv_type,
                                             attack_eps_value=attack_eps,
                                             steps=adv_steps,
                                             num_attacks=num_attacks,
                                             dataset_name=dataset_name)
            adv_accuracies.append(adv_accuracy)
        return nat_acc, adv_accuracies


def run_adv_rob_exp(training_logdir,
                    exp_logdir,
                    device,
                    resnet_depth = 110,
                    clf_epochs = 150,
                    vae_epochs = 150,
                    use_step_lr = True,
                    lr_schedule_step = 50,
                    lr_schedule_gamma = .1,
                    train_set=None,
                    test_set=None):
    sw = SummaryWriter(exp_logdir)
    linf_norms = [1 / 255, 2 / 255, 4 / 255, 8 / 255]
    smoothing_sigmas = [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255, 32 / 255, 64 / 255, 128 / 255]
    exp = AdvRobustnessNaturalTraining(training_logdir=training_logdir,
                                       exp_logdir=exp_logdir,
                                       device=device,
                                       train_set=train_set,
                                       test_set=test_set)
    nat_acc, adv_accs = exp.adv_rob_base_clf(net_depth=resnet_depth,
                         clf_epochs=clf_epochs,
                         adv_norms=linf_norms,
                         adv_type='linf',
                         use_step_lr=use_step_lr,
                         lr_schedule_gamma=lr_schedule_gamma,
                         lr_schedule_step=lr_schedule_step)
    write_res(sw, nat_acc, adv_accs, linf_norms, "resnet110_vanilla", 'linf')
    for sigma in smoothing_sigmas:
        nat_acc, adv_accs = exp.adv_rob_smoothclf(net_depth=resnet_depth,
                              clf_epochs=clf_epochs,
                              adv_norms=linf_norms,
                              adv_type='linf',
                              smoothing_sigma=sigma,
                              use_step_lr=use_step_lr,
                              lr_schedule_gamma=lr_schedule_gamma,
                              lr_schedule_step=lr_schedule_step)
        write_res(sw, nat_acc, adv_accs, linf_norms, f"resnet110smooth_{round(sigma, 3)}", 'linf')

    for smooth_vae_type in ['sample', 'latent']:
        for sigma in smoothing_sigmas:
            nat_acc, adv_accs = exp.adv_rob_smoothvae_clf(clf_epochs=clf_epochs,
                                      vae_epochs=vae_epochs,
                                      net_depth=resnet_depth,
                                      smoothing_vae_sigma=sigma,
                                      adv_norms=linf_norms,
                                      adv_type='linf',
                                      smooth_vae_version=smooth_vae_type,
                                      use_step_lr=use_step_lr,
                                      lr_schedule_gamma=lr_schedule_gamma,
                                      lr_schedule_step=lr_schedule_step)
            write_res(sw, nat_acc, adv_accs, linf_norms, f"resnet110smoothVAE_{smooth_vae_type}_{round(sigma, 3)}", 'linf')
    return


def write_res(sw : SummaryWriter, nat_acc, adv_accs, norms, model_name, adv_type):
    metric_dct = {'Nat Acc' : nat_acc}
    for i, eps in enumerate(norms):
        label = round(eps, 3)
        metric_dct[f"{adv_type}_norm_{label}"] = adv_accs[i]
    sw.add_hparams({'model' : model_name}, metric_dct)


