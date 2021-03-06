import os
import time
import datetime
import yaml
import paddle
from paddle.io import DataLoader
from dataset import CityscapesDataset
from models import ICNet
from utils import ICNetLoss, SegmentationMetric, SetupLogger


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        # self.device = 'gpu'
        self.dataparallel = False

        # dataset and dataloader
        train_dataset = CityscapesDataset(root=cfg["train"]["cityscapes_root"],
                                          split='train',
                                          base_size=cfg["model"]["base_size"],
                                          crop_size=cfg["model"]["crop_size"])
        val_dataset = CityscapesDataset(root=cfg["train"]["cityscapes_root"],
                                        split='val',
                                        base_size=cfg["model"]["base_size"],
                                        crop_size=cfg["model"]["crop_size"])
        self.train_dataloader = DataLoader(dataset=train_dataset,
                                           batch_size=cfg["train"]["train_batch_size"],
                                           shuffle=True,
                                           num_workers=4,
                                           drop_last=False)
        self.val_dataloader = DataLoader(dataset=val_dataset,
                                         batch_size=cfg["train"]["valid_batch_size"],
                                         shuffle=False,
                                         num_workers=4,
                                         drop_last=False)

        self.iters_per_epoch = len(self.train_dataloader)
        self.max_iters = cfg["train"]["epochs"] * self.iters_per_epoch

        # create network
        self.model = ICNet(nclass=train_dataset.NUM_CLASS, backbone=cfg["model"]["backbone"])

        # create criterion
        self.criterion = ICNetLoss(ignore_index=train_dataset.IGNORE_INDEX)

        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        self.lr_scheduler = paddle.optimizer.lr.PolynomialDecay(
            learning_rate=cfg["optimizer"]["init_lr"],
            decay_steps=self.max_iters,
            end_lr=0.0,
            power=0.9,
            cycle=False
        )
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'learning_rate': 1.0})
            print('lr*1')
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                print(getattr(self.model, module).parameters())
                params_list.append({'params': getattr(self.model, module).parameters(), 'learning_rate': 10.0})
                print("lr*10")
        self.optimizer = paddle.optimizer.Momentum(parameters=params_list,
                                                    learning_rate=self.lr_scheduler,
                                                    momentum=cfg["optimizer"]["momentum"],
                                                    weight_decay=cfg["optimizer"]["weight_decay"])


        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.NUM_CLASS)

        self.current_mIoU = 0.0
        self.best_mIoU = 0.0

        self.epochs = cfg["train"]["epochs"]
        self.current_epoch = 0
        self.current_iteration = 0

    def train(self):
        epochs, max_iters = self.epochs, self.max_iters
        log_per_iters = self.cfg["train"]["log_iter"]
        val_per_iters = self.cfg["train"]["val_epoch"] * self.iters_per_epoch

        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()

        for _ in range(self.epochs):
            self.current_epoch += 1
            lsit_pixAcc = []
            list_mIoU = []
            list_loss = []
            self.metric.reset()
            for i, (images, targets, _) in enumerate(self.train_dataloader()):
                self.current_iteration += 1
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                self.metric.update(outputs[0], targets)
                pixAcc, mIoU = self.metric.get()
                lsit_pixAcc.append(pixAcc)
                list_mIoU.append(mIoU.item())
                list_loss.append(loss.item())

                self.optimizer.clear_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                eta_seconds = ((time.time() - start_time) / self.current_iteration) * (
                            max_iters - self.current_iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if self.current_iteration % log_per_iters == 0:
                    logger.info(
                        "Epochs: {}/{} || Iters: {}/{} || Lr: {} || Loss: {} || mIoU: {} || Cost Time: {} || Estimated Time: {}".format(
                            self.current_epoch, self.epochs,
                            self.current_iteration, max_iters,
                            self.optimizer.get_lr(),
                            loss.item(),
                            mIoU.item(),
                            str(datetime.timedelta(seconds=int(time.time() - start_time))),
                            eta_string))

            average_pixAcc = sum(lsit_pixAcc) / len(lsit_pixAcc)
            average_mIoU = sum(list_mIoU) / len(list_mIoU)
            average_loss = sum(list_loss) / len(list_loss)
            logger.info(
                "Epochs: {}/{}, Average loss: {}, Average mIoU: {}, Average pixAcc: {}".format(self.current_epoch,
                                                                                               self.epochs,
                                                                                               average_loss,
                                                                                               average_mIoU,
                                                                                               average_pixAcc))

            if self.current_iteration % val_per_iters == 0:
                self.validation()
                self.model.train()

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self):
        is_best = False
        self.metric.reset()
        model = self.model
        model.eval()
        lsit_pixAcc = []
        list_mIoU = []
        list_loss = []
        for i, (image, targets, filename) in enumerate(self.val_dataloader):
            with paddle.no_grad():
                outputs = model(image)
                loss = self.criterion(outputs, targets)
            self.metric.update(outputs[0], targets)
            pixAcc, mIoU = self.metric.get()
            lsit_pixAcc.append(pixAcc)
            list_mIoU.append(mIoU.item())
            list_loss.append(loss.item())

        average_pixAcc = sum(lsit_pixAcc) / len(lsit_pixAcc)
        average_mIoU = sum(list_mIoU) / len(list_mIoU)
        average_loss = sum(list_loss) / len(list_loss)
        self.current_mIoU = average_mIoU
        logger.info(
            "Validation: Average loss: {}, Average mIoU: {}, Average pixAcc: {}".format(average_loss, average_mIoU,
                                                                                        average_pixAcc))

        if self.current_mIoU > self.best_mIoU:
            is_best = True
            self.best_mIoU = self.current_mIoU
        if is_best:
            save_checkpoint(self.model, self.cfg, self.current_epoch, is_best, self.current_mIoU, self.dataparallel)


def save_checkpoint(model, cfg, epoch=0, is_best=False, mIoU=0.0, dataparallel=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(cfg["train"]["ckpt_dir"])
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}_{}.pdparams'.format(cfg["model"]["name"], cfg["model"]["backbone"], epoch, mIoU)
    filename = os.path.join(directory, filename)
    if dataparallel:
        model = model.module
    if is_best:
        best_filename = '{}_{}_{}_{}_best_model.pdparams'.format(cfg["model"]["name"], cfg["model"]["backbone"], epoch,
                                                                 mIoU)
        best_filename = os.path.join(directory, best_filename)
        paddle.save(model.state_dict(), best_filename)


if __name__ == '__main__':
    # Set config file
    config_path = "./configs/icnet.yaml"
    with open(config_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file.read())
        print(cfg)
        print(cfg["model"]["backbone"])
        print(cfg["train"]["specific_gpu_num"])

    # Use specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["train"]["specific_gpu_num"])
    num_gpus = len(cfg["train"]["specific_gpu_num"].split(','))

    # Set logger
    logger = SetupLogger(name="semantic_segmentation",
                         save_dir=cfg["train"]["ckpt_dir"],
                         distributed_rank=0,
                         filename='{}_{}_log.txt'.format(cfg["model"]["name"], cfg["model"]["backbone"]))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    # Start train
    trainer = Trainer(cfg)
    trainer.train()

