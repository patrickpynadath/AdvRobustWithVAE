import torch
import torch.distributions.normal as normal_dist
import torch.distributions.uniform as uniform_dist
import numpy as np


# Code from https://gitlab.com/speech-interaction-technology-aalto-university/nsvq Paper:
# https://ieeexplore.ieee.org/abstract/document/9696322 Write up:
# https://towardsdatascience.com/improving-vector-quantization-in-vector-quantized-variational-autoencoders-vq-vae
# -915f5814b5ce
# Motivation: For some reason, the perplexity during training of a VQ-VAE with the standard STE for the
# typical VQ approach was not stable, and hence reproducibility of results was a big concern. This alternative
# approach to VQ is vastly more stable, converges quicker, and is thus more reproducible.

class NSVQ(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_epochs, num_training_samples, batch_size, device,
                 discarding_threshold=0.01, num_first_cbr=10, first_cbr_coefficient=0.005, second_cbr_coefficient=0.03,
                 initialization='normal'):
        super(NSVQ, self).__init__()

        """
        Inputs:

        1. num_embeddings = Number of codebook entries

        2. embedding_dim = Embedding dimension (dimensionality of each input data sample or codebook entry)

        3. num_epochs = Total number of epochs for training (one epoch corresponds to fetching whole train set)

        4. num_training_samples = Total number of samples in train set

        5. batch_size = Number of data samples in one training batch

        6. device = The device which executes the code (CPU or GPU)

        ########## change the following inputs based on your application ##########

        7. discarding_threshold = Percentage threshold for discarding unused codebooks

        8. num_first_cbr = Number of times to perform codebook replacement function in the first codebook replacement 
                            period                
        9. first_cbr_coefficient = Coefficient which determines num of batches for the first codebook replacement cycle

        10. second_cbr_coefficient = Coefficient which determines num of batches for the second codebook replacement cycle

        11. initialization = Initial distribution for codebooks
        """

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.batch_counter = 0
        self.eps = 1e-12
        self.device = device

        self.num_epochs = num_epochs
        self.num_training_samples = num_training_samples
        self.batch_size = batch_size

        self.num_total_updates = np.floor((num_training_samples / batch_size) * num_epochs)
        self.first_cbr_cycle = np.floor(first_cbr_coefficient * self.num_total_updates)
        self.first_cbr_period = num_first_cbr * self.first_cbr_cycle
        self.second_cbr_cycle = np.floor(second_cbr_coefficient * self.num_total_updates)
        self.discarding_threshold = discarding_threshold
        # Index of transition batch between first and second codebook replacement periods
        self.transition_value = (np.ceil(self.first_cbr_period / self.second_cbr_cycle) + 1) * self.second_cbr_cycle

        if initialization == 'normal':
            codebooks = torch.randn(self.num_embeddings, self.embedding_dim, device=device)

        elif initialization == 'uniform':
            codebooks = uniform_dist.Uniform(-1 / self.num_embeddings, 1 / self.num_embeddings).sample(
                [self.num_embeddings, self.embedding_dim])

        else:
            raise ValueError("initialization should be one of the 'normal' and 'uniform' strings")

        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)

        # Counter variable which contains number of times each codebook is used
        self.codebooks_used = torch.zeros(self.num_embeddings, device=device)

    def forward(self, inputs):

        """
        This function performs the main proposed vector quantization function.

        N: number of input data samples
        K: num_embeddings (number of codebook entries)
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        input: input_data (input data matrix which is going to be vector quantized | shape: (NxD) )
        output: quantized_input (vector quantized version of input data matrix | shape: (NxD) )
        """
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # compute the distances between input and codebooks vectors
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     - 2 * (torch.matmul(flat_input, self.codebooks.t()))
                     + torch.sum(self.codebooks.t() ** 2, dim=0, keepdim=True))

        min_indices = torch.argmin(distances, dim=1)

        best_entries = self.codebooks[min_indices]
        random_vector = normal_dist.Normal(0, 1).sample(flat_input.shape).to(self.device)

        norm_best_entries = (flat_input - best_entries).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()

        # defining vector quantization error
        vq_error = (norm_best_entries / norm_random_vector + self.eps) * random_vector

        quantized_input = flat_input + vq_error
        quantized_input = quantized_input.view(input_shape)

        # calculating perplexity

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        with torch.no_grad():
            self.codebooks_used[min_indices] += 1
            self.batch_counter += 1

        # Checking whether it is the time to perform codebook replacement
        if ((self.batch_counter % self.first_cbr_cycle == 0) & (self.batch_counter <= self.first_cbr_period)):
            self.replace_unused_codebooks(self.first_cbr_cycle)

        if self.batch_counter == self.transition_value:
            self.replace_unused_codebooks(np.remainder(self.transition_value, self.first_cbr_period))

        if ((self.batch_counter % self.second_cbr_cycle == 0) &
                (self.transition_value < self.batch_counter <= self.num_total_updates - self.second_cbr_cycle)):
            self.replace_unused_codebooks(self.second_cbr_cycle)

        return quantized_input.permute(0, 3, 1, 2).contiguous(), perplexity

    def replace_unused_codebooks(self, num_batches):

        """
        This function gets the number of batches as input and replaces the codebooks which are used less than a
        threshold percentage (discarding_threshold) during these number of batches with the active (used) codebooks

        input: num_batches (number of batches during which we determine whether the codebook is used or unused)
        """

        with torch.no_grad():

            unused_indices = torch.where((self.codebooks_used.cpu() / num_batches) < self.discarding_threshold)[0]
            used_indices = torch.where((self.codebooks_used.cpu() / num_batches) >= self.discarding_threshold)[0]

            unused_count = unused_indices.shape[0]
            used_count = used_indices.shape[0]

            if used_count == 0:
                print(f'####### used_indices equals zero / shuffling whole codebooks ######')
                self.codebooks += self.eps * torch.randn(self.codebooks.size(), device=self.device).clone()
            else:
                used = self.codebooks[used_indices].clone()
                if used_count < unused_count:
                    used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                    used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                else:
                    used_codebooks = used

                self.codebooks[unused_indices] *= 0
                self.codebooks[unused_indices] += used_codebooks[range(unused_count)] + self.eps * torch.randn(
                    (unused_count, self.embedding_dim), device=self.device).clone()

            print(f'************* Replaced ' + str(unused_count) + f' codebooks *************')
            self.codebooks_used[:] = 0.0