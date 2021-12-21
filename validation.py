import torch
import time
import sys
import mlflow
import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy, \
    calculate_precision_recall_fscore, calculate_auc


def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              logger,
              tb_writer=None,
              distributed=False,
              use_mlflow=False,
              precision_recall_fscore=False,
              auc=False):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    with torch.no_grad():
        all_targets = []
        all_outputs = []
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if precision_recall_fscore:
                all_targets.append(targets)
                all_outputs.append(outputs)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.\
                format(epoch,
                       i + 1,
                       len(data_loader),
                       batch_time=batch_time,
                       data_time=data_time,
                       loss=losses,
                       acc=accuracies)
                  )

    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    if precision_recall_fscore:
        precision, recall, fscore = calculate_precision_recall_fscore(
            torch.cat(all_outputs, dim=0),
            torch.cat(all_targets, dim=0),
            pos_label=1)

    if auc:
        auc_val = calculate_auc(torch.cat(all_outputs, dim=0),
                                torch.cat(all_targets, dim=0),
                                pos_label=1)

    if logger is not None:
        log_data = {'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg}
        if precision_recall_fscore:
            log_data["precision"] = precision
            log_data["recall"] = recall
            log_data["fscore"] = fscore
        if auc:
            log_data["auc"] = auc_val
        logger.log(log_data)

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/acc', accuracies.avg, epoch)
        if precision_recall_fscore:
            tb_writer.add_scalar('val/precision', precision, epoch)
            tb_writer.add_scalar('val/recall', recall, epoch)
            tb_writer.add_scalar('val/fscore', fscore, epoch)
        if auc:
            tb_writer.add_scalar('val/auc', auc_val, epoch)

    if use_mlflow:
        metrics = {"val/loss": losses.avg, "val/acc": accuracies.avg}
        if precision_recall_fscore:
            metrics["val/precision"] = precision
            metrics["val/recall"] = recall
            metrics["val/fscore"] = fscore
        if auc:
            metrics["val/auc"] = auc_val
        mlflow.log_metrics(metrics=metrics, step=epoch)

    return losses.avg
