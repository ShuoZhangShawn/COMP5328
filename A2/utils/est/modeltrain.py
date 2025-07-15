import torch
from utils.est.help import AverageMeter
from utils.est.help import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch, model, optimizer, criterion, train_loader):
    """Training a pytorch nn_model."""
    top1_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    # swith model to to train mode
    model.train()
    for step, (data, targets) in enumerate(train_loader):
        # prepare min_batch
        data = data.to(device)
        targets = targets.to(device)

        # predict
        preds = model(data)

        # forward
        loss = criterion(preds, targets)

        # set all gradients to zero
        optimizer.zero_grad()

        # backward
        loss.backward()

        # update all gradients
        optimizer.step()

        # calculate accuracy
        [top1_acc] = accuracy(preds.data, targets.data, topk=(1,))
        # record accuary and cross entropy losss
        min_batch_size = data.size(0)
        top1_acc_meter.update(top1_acc.item(), min_batch_size)
        loss_meter.update(loss.item(), min_batch_size)

    print("Train epoch ",epoch," Accuracy ",top1_acc_meter.avg)