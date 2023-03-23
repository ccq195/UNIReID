import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    arguments2 = {}
    arguments2["num_epoch"] = num_epoch
    arguments2["iteration"] = 0

    arguments3 = {}
    arguments3["num_epoch"] = num_epoch
    arguments3["iteration"] = 0

    arguments4 = {}
    arguments4["num_epoch"] = num_epoch
    arguments4["iteration"] = 0

    logger = logging.getLogger("CLIP2ReID.train")
    logger.info('start training')

    loss_meter = AverageMeter()
    mcm_loss_meter = AverageMeter()
    mlm_loss_meter = AverageMeter()
    mcq_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_ttop1 = 0.0
    best_stop1 = 0.0
    best_itop1 = 0.0
    best_ftop1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        mcm_loss_meter.reset()
        mlm_loss_meter.reset()
        mcq_loss_meter.reset()
        model.train()
        
        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            loss_meter.update(total_loss.item(), batch['images'].shape[0])
            acc_meter.update(ret.get('acc', 0), 1)
            
            mcm_loss_meter.update(ret.get('mcm_loss', 0), batch['images'].shape[0])
            mlm_loss_meter.update(ret.get('mlm_loss', 0), batch['images'].shape[0])
            mcq_loss_meter.update(ret.get('mcq_loss', 0), batch['images'].shape[0])

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                logger.info(
                    f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}] Loss: {loss_meter.avg:.4f}, mcm_loss: {mcm_loss_meter.avg:.4f}, mcq_loss: {mcq_loss_meter.avg:.4f}, mlm_loss: {mlm_loss_meter.avg:.4f}, Acc: {acc_meter.avg:.3f}, Base Lr: {scheduler.get_lr()[0]:.2e}"
                )

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        tb_writer.add_scalar('loss', loss_meter.avg, epoch)
        tb_writer.add_scalar('mcm_loss', mcm_loss_meter.avg, epoch)
        tb_writer.add_scalar('mlm_loss', mlm_loss_meter.avg, epoch)
        tb_writer.add_scalar('mcq_loss', mcq_loss_meter.avg, epoch)
        tb_writer.add_scalar('acc', acc_meter.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    ttop1, stop1, itop1 = evaluator.eval(model.module.eval())
                ttop1, stop1, itop1 = evaluator.eval(model.eval())

                ftop1 = (ttop1 + stop1 + itop1)/3.0

                torch.cuda.empty_cache()
                if best_ttop1 < ttop1:
                    best_ttop1 = ttop1
                    arguments["epoch"] = epoch
                    checkpointer.save("text_best", **arguments)

                if best_stop1 < stop1:
                    best_stop1 = stop1
                    arguments2["epoch"] = epoch
                    checkpointer.save("sketch_best", **arguments)

                if best_itop1 < itop1:
                    best_itop1 = itop1
                    arguments3["epoch"] = epoch
                    checkpointer.save("fusion_best", **arguments)

                if best_ftop1 < ftop1:
                    best_ftop1 = ftop1
                    arguments4["epoch"] = epoch
                    checkpointer.save("average_best", **arguments)

    logger.info(f"text best R1: {best_ttop1} at epoch {arguments['epoch']}")
    logger.info(f"sketch best R1: {best_stop1} at epoch {arguments2['epoch']}")
    logger.info(f"fusion best R1: {best_itop1} at epoch {arguments3['epoch']}")
    logger.info(f"average best R1: {best_ftop1} at epoch {arguments4['epoch']}")


def do_inference(args, model, test_img_loader, test_txt_loader, test_sketch_loader):

    logger = logging.getLogger("CLIP2ReID.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(args, test_img_loader, test_txt_loader, test_sketch_loader)
    ttop1, stop1, itop1 = evaluator.eval(model.eval())
    # top1 = evaluator.eval_by_proj(model.eval())

    # table = PrettyTable(["task", "R1", "R5", "R10", "mAP"])
    # table.float_format = '.4'
    # table.add_row(['t2i', cmc[0], cmc[4], cmc[9], mAP])
    # logger.info("Validation Results: ")
    # logger.info('\n' + str(table))
