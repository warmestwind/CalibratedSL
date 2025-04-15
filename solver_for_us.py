from torch.utils.tensorboard import SummaryWriter
from model import *
from loss import Loss
from util import summary
from loss.loss import EviLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

class Operator:
    def __init__(self, config, ckeck_point):
        self.config = config
        self.epochs = config.epochs
        self.uncertainty = config.uncertainty
        self.ckpt = ckeck_point
        self.tensorboard = config.tensorboard
        if self.tensorboard:
            self.summary_writer = SummaryWriter(self.ckpt.log_dir, 300)
        self.T = 1

        # set model, criterion, optimizer
        self.model = Model(config)
        summary(self.model, config_file=self.ckpt.config_file)



        self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-7)  # T_max是学习率周期

        self.evi = EviLoss()

        # load ckpt, model, optimizer
        if self.ckpt.exp_load is not None or not config.is_train:
            print("Loading model... ")
            self.load(self.ckpt)
            print(self.ckpt.last_epoch, self.ckpt.global_step)



    def train(self, data_loader):
        last_epoch = self.ckpt.last_epoch
        print(last_epoch)
        train_batch_num = len(data_loader['train'])

        for epoch in range(last_epoch, self.epochs):
            for batch_idx, batch_data in enumerate(data_loader['train']):
                batch_input, batch_label, batch_coords, batch_indexes, batch_input_n= batch_data
                batch_input_n = batch_input_n.float().to(self.config.device)
                batch_input = batch_input.float().to(self.config.device)
                batch_label = batch_label.float().to(self.config.device)
                batch_coords = batch_coords.float().to(self.config.device)
                # batch_indexes = batch_indexes.float().to(self.config.device)

                # forward

                results = self.model(batch_input)

                loss_p, loss_u, loss_dce = self.evi(results, batch_label)

                loss_total = loss_u+loss_p+loss_dce

                # backward
                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                print('Epoch: {:03d}/{:03d}, Iter: {:03d}/{:03d}, Loss: {:.2f},  {:.2f}, {:.2f}, {:.2f}'
                      .format(epoch, self.config.epochs,
                              batch_idx, train_batch_num,
                              loss_total.item(),  loss_p.item(), loss_u.item(), loss_dce.item()))

            # use tensorboard
            if self.tensorboard:
                current_global_step = epoch # self.ckpt.step()
                self.summary_writer.add_scalar('train/loss',
                                               loss_total, current_global_step)
                self.summary_writer.add_scalar('train/loss_p',
                                               loss_p, current_global_step)
                self.summary_writer.add_scalar('train/loss_u',
                                               loss_u, current_global_step)
                self.summary_writer.add_scalar('train/loss_dce',
                                               loss_dce, current_global_step)

            self.scheduler.step()
            if epoch%100==0 or epoch==self.epochs-1:
                self.save(self.ckpt, epoch)
                # self.test(data_loader)
                self.model.train()

        self.summary_writer.close()


    def load(self, ckpt):
        ckpt.load() # load ckpt
        self.model.load(ckpt) # load model


    def save(self, ckpt, epoch):
        ckpt.save(epoch) # save ckpt: global_step, last_epoch
        self.model.save(ckpt, epoch) # save model: weight


