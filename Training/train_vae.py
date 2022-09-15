from torch import optim
from torch.autograd import Variable
from Models import vae_models
from tqdm import tqdm

# source: https://github.com/SashaMalysheva/Pytorch-VAE
def train_vae(model, data_loader, epochs=10,
              batch_size=32, lr=3e-04, weight_decay=1e-5, device='cuda'):
    # prepare optimizer and model
    model.train()
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay,
    )
    len_dataset = len(data_loader.dataset)

    epoch_start = 1

    for epoch in range(epoch_start, epochs+1):
        data_stream = tqdm(enumerate(data_loader, 1), total=len(data_loader),  position=0, leave=True)

        for batch_index, (x, _) in data_stream:
            # where are we?
            iteration = (epoch-1)*(len_dataset//batch_size) + batch_index

            # prepare data on gpu if needed
            x = Variable(x).to(device)

            # flush gradients and run the model forward
            optimizer.zero_grad()
            out = model(x)
            loss = model.loss_function(out)
            l = loss['loss'].item()
            loss['loss'].backward()
            optimizer.step()


            # update progress
            data_stream.set_description((
                'epoch: {epoch} | '
                'iteration: {iteration} | '
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                'loss => '
                'total: {total_loss:.4f} / '
            ).format(
                epoch=epoch,
                iteration=iteration,
                trained=batch_index * len(x),
                total=len(data_loader.dataset),
                progress=(100. * batch_index / len(data_loader)),
                total_loss=l
            ))
