import torch
from utils.est.help import AverageMeter
from utils.est.help import accuracy
from utils.est.Dataloader import get_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(epoch, model, criterion, test_loader, is_test=False):
    """testing of a nn_model."""
    top1_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    # swith model to to eval mode
    model.eval()
    for step, (data, targets) in enumerate(test_loader):
        # prepare min_batch
        data = data.to(device)
        targets = targets.to(device)

        # predict
        with torch.no_grad(): #禁用梯度计算，我们不需要再更新模型参数了
            preds = model(data)

        # forward
        loss = criterion(preds, targets)
            
        # calculate accuracy
        [top1_acc] = accuracy(preds.data, targets.data, topk=(1,))
  
        # record accuary and cross entropy losss
        min_batch_size = data.size(0)
        top1_acc_meter.update(top1_acc.item(), min_batch_size)
        loss_meter.update(loss.item(), min_batch_size)

    top1_acc_avg = top1_acc_meter.avg
    if(is_test == False):
        print("Validate epoch ",epoch," Accuracy ",top1_acc_avg)
    else:
        print("Test epoch ",epoch," Accuracy ",top1_acc_avg)

    return top1_acc_avg
