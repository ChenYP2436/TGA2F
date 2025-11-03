from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment, adjustment_len

from sklearn.preprocessing import StandardScaler
import torch.multiprocessing
from utils.merge_preds import merge_preds
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

from layers.fre_rec_loss import frequency_loss, frequency_criterion
from utils.save_plot import (recon_labels, max_per_row,  write_metrics, plot_label, log_print , find_max_fscore)

warnings.filterwarnings('ignore')
from model import cat_train

from model import iTransformerGuide



def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class Exp_Anomaly_Prediction():
    def __init__(self, args):
        self.guide_model_dict = {
            'iTransGuide': iTransformerGuide,
        }
        self.args = args
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.auxi_loss = frequency_loss(self.args)
        self.guide_model = self.guide_model_dict[self.args.guide_model].guide_model(self.args).float().to(self.device)
        self.model = self._build_model()

    def _build_model(self):
        self.model = cat_train.Model(self.args).float().to(self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.args.device_ids)
        return self.model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def time_freq_mae(self, batch_y, outputs):
        # time mae loss
        t_loss = ((outputs - batch_y)**2).mean()

        # freq mae loss
        f_loss = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
        f_loss = f_loss.abs().mean()

        return (1 - self.args.pred_alpha) * t_loss + self.args.pred_alpha * f_loss

    def MSE_weighted(self, batch_y, outputs, weights):
        t_loss_per_point = (outputs - batch_y) ** 2
        loss = (weights * t_loss_per_point).mean()

        return loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                pred, rec, dcloss = self.model(batch_x)

                self.guide_model.eval()
                with torch.no_grad():
                    rec_batch_y = self.guide_model(batch_y)  # [batch, seq_len, dim]

                recon_loss_per_point = torch.abs(batch_y - rec_batch_y)  # [batch, seq_len, dim]
                tau = 0.1
                weights = torch.softmax(recon_loss_per_point / tau, dim=1)
                loss_pred = self.MSE_weighted(batch_y, pred, weights).detach().cpu().numpy()

                original_seq = torch.cat([batch_x, pred], dim=1)
                original_seq_history = original_seq[:, :self.args.seq_len, :]
                original_seq_pred = original_seq[:, self.args.seq_len:, :]
                rec_history = rec[:, :self.args.seq_len, :]
                rec_pred = rec[:, self.args.seq_len:, :]
                rec_history_t_loss = self.criterion(original_seq_history, rec_history)
                rec_perd_t_loss = self.criterion(original_seq_pred, rec_pred)
                rec_t_loss = 0 * rec_history_t_loss + 1 * rec_perd_t_loss
                rec_history_f_loss = self.auxi_loss(original_seq_history, rec_history)
                rec_pred_f_loss = self.auxi_loss(original_seq_pred, rec_pred)
                rec_f_loss = 0 * rec_history_f_loss + 1 * rec_pred_f_loss
                loss_recon = rec_t_loss + self.args.auxi_lambda * rec_f_loss

                with torch.no_grad():
                    rec_guide = self.guide_model(pred)

                guide_loss = self.criterion(rec_pred, rec_guide)
                loss_catch = loss_recon + self.args.guide_lambda * guide_loss + self.args.dc_lambda * dcloss

                loss_detec = loss_catch.detach().cpu().numpy()
                loss = loss_pred + self.args.detec_lambda * loss_detec
                total_loss.append(loss)

            total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints + self.args.data + '/' + self.args.model + '/', setting)
        if not os.path.exists(path):
            os.makedirs(path)

        result_path = './results/' + self.args.data + '/' + self.args.model + '/' + setting + '/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if self.args.use_guide:
            if self.args.train_guide:
                guide_model_path = './checkpoints_guide/' + self.args.data + '/' + self.args.guide_model
                if not os.path.exists(guide_model_path):
                    os.makedirs(guide_model_path)
                self.train_guide(train_data, train_loader, vali_data, vali_loader, result_path, guide_model_path)
            else:
                guide_model_path = './checkpoints_guide/' + self.args.data + '/' + self.args.guide_model + '/'+ 'checkpoint_guide' + '_' + str(self.args.seq_len) + '.pth'
                self.guide_model.load_state_dict(torch.load(guide_model_path))

        # 打印所有参数
        for arg, value in vars(self.args).items():
            log_print(result_path, "{}: {}".format(arg,  value))

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log_print(result_path, f"Total trainable parameters: {total_params}")
        log_print(result_path, "\n==== Training model ====\n")


        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)


        main_params = [param for name, param in self.model.named_parameters() if 'mask_generator' not in name]
        optimizer = torch.optim.Adam(main_params, self.args.lr)
        optimizerM = torch.optim.Adam(self.model.detection_layer.mask_generator.parameters(), lr=self.args.Mlr)

        iter_count = 0

        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):

            train_loss = []

            self.model.train()
            epoch_time = time.time()
            step = min(int(len(train_loader) / 10), 100)
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1

                optimizer.zero_grad()
                optimizerM.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred, rec, dcloss = self.model(batch_x)

                if self.args.use_guide:
                    self.guide_model.eval()
                    with torch.no_grad():
                        rec_batch_y = self.guide_model(batch_y)  # [batch, seq_len, dim]

                    recon_loss_per_point = torch.abs(batch_y - rec_batch_y)  # [batch, seq_len, dim]
                    tau = 0.1
                    weights = torch.softmax(recon_loss_per_point / tau, dim=1)
                    loss_pred = self.MSE_weighted(batch_y, pred, weights)
                else:
                    loss_pred = self.time_freq_mae(batch_y, pred)

                original_seq = torch.cat([batch_x, pred], dim=1)
                original_seq_history = original_seq[:, :self.args.seq_len, :]
                original_seq_pred = original_seq[:, self.args.seq_len:, :]
                rec_history = rec[:, :self.args.seq_len, :]
                rec_pred = rec[:, self.args.seq_len:, :]
                rec_history_t_loss = self.criterion(original_seq_history, rec_history)
                rec_perd_t_loss = self.criterion(original_seq_pred, rec_pred)
                rec_t_loss = 0 * rec_history_t_loss + 1 * rec_perd_t_loss
                rec_history_f_loss = self.auxi_loss(original_seq_history, rec_history)
                rec_pred_f_loss = self.auxi_loss(original_seq_pred, rec_pred)
                rec_f_loss = 0 * rec_history_f_loss + 1 * rec_pred_f_loss
                loss_recon = rec_t_loss + self.args.auxi_lambda * rec_f_loss

                with torch.no_grad():
                    rec_guide = self.guide_model(pred)

                guide_loss = self.criterion(rec_pred, rec_guide)
                loss_catch = loss_recon + self.args.guide_lambda * guide_loss + self.args.dc_lambda * dcloss

                loss_total = loss_pred + loss_catch * self.args.detec_lambda
                loss_total.backward()
                optimizer.step()

                if (i + 1) % step == 0:
                    optimizerM.step()

                loss_detec = loss_catch
                loss = loss_pred + loss_catch * self.args.detec_lambda

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    log_print(result_path,
                        "\t iters: {0}, epoch: {1} | training loss: {2:.7f} | training pred loss: {3:.7f} | training recon loss: {4:.7f}".format(
                            i + 1, epoch + 1, loss, loss_pred, loss_detec
                        )
                    )

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    log_print(result_path, "\t speed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            log_print(result_path, "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            log_print(result_path, "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                log_print(result_path, "Early stopping")
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali_guide(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.guide_model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                rec = self.guide_model(batch_x)
                loss = self.criterion(batch_x, rec)
                total_loss.append(loss.item())
            total_loss = np.mean(total_loss)
        self.guide_model.train()

        return total_loss

    def train_guide(self, train_data, train_loader, vali_data, vali_loader, result_path, model_path):
        log_print(result_path, 'Start training guide model...')
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        guide_params = self.guide_model.parameters()
        guide_optimizer = optim.Adam(guide_params, lr=self.args.lr)
        criterion = self._select_criterion()
        self.guide_model.train()
        train_steps = len(train_loader)
        time_now = time.time()
        iter_count = 0
        for epoch in range(self.args.train_epochs):
            train_loss = []
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                guide_optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                rec = self.guide_model(batch_x)
                loss = self.criterion(batch_x, rec)
                train_loss.append(loss.item())

                loss.backward()
                guide_optimizer.step()

                if (i + 1) % 100 == 0:
                    log_print(result_path,"\t iters: {0}, epoch: {1} | training loss: {2:.7f}".format(i + 1, epoch + 1, loss))

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    log_print(result_path, "\t speed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            log_print(result_path, "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali_guide(vali_data, vali_loader, criterion)

            log_print(result_path, "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.guide_model, model_path, 'checkpoint_guide' + '_' + str(self.args.seq_len) + '.pth')
            if early_stopping.early_stop:
                log_print(result_path, "Early stopping")
                break

            adjust_learning_rate(guide_optimizer, epoch + 1, self.args)

        best_model_path = model_path + '/' + 'checkpoint_guide' + '_' + str(self.args.seq_len) + '.pth'
        self.guide_model.load_state_dict(torch.load(best_model_path))


    def test(self, setting, test=0):
        train_data, train_loader = self._get_data(flag='train')
        thre_data, thre_loader = self._get_data(flag='thre')


        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + self.args.data + '/' + self.args.model + '/' + setting, 'checkpoint.pth')))

        result_path = './results/' + self.args.data + '/' + self.args.model + '/' + setting + '/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        self.model.eval()

        print("---------The data and model have been loaded---------")
        self.criterion = nn.MSELoss(reduce=False)
        self.temp_anomaly_criterion = nn.MSELoss(reduce=False)
        self.freq_anomaly_criterion = frequency_criterion(self.args)
        attens_energy = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                pred, rec, dcloss = self.model(batch_x)


                original_seq = torch.cat([batch_x, pred], dim=1)
                original_seq_history = original_seq[:, :self.args.seq_len, :]
                original_seq_pred = original_seq[:, self.args.seq_len:, :]
                rec_history = rec[:, :self.args.seq_len, :]
                rec_pred = rec[:, self.args.seq_len:, :]
                history_time_score = torch.mean(self.temp_anomaly_criterion(original_seq_history, rec_history), dim=-1)
                pred_time_score = torch.mean(self.temp_anomaly_criterion(original_seq_pred, rec_pred), dim=-1)
                time_score = 0 * history_time_score + 1 * pred_time_score
                history_freq_score = torch.mean(self.freq_anomaly_criterion(original_seq_history, rec_history), dim=-1)
                pred_freq_score = torch.mean(self.freq_anomaly_criterion(original_seq_pred, rec_pred), dim=-1)
                freq_score = 0 * history_freq_score + 1 * pred_freq_score
                score = (time_score + self.args.score_lambda * freq_score).detach().cpu().numpy()

                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0)
        train_energy = merge_preds(attens_energy, stride=self.args.step)
        print("---------The forward propagation of the training set is complete---------")

        attens_energy = []
        real_labels = []
        preds_series = []
        preds_series_rec = []
        true_series = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_y_label) in enumerate(thre_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                pred, rec, dcloss = self.model(batch_x)

                original_seq = torch.cat([batch_x, pred], dim=1)
                original_seq_history = original_seq[:, :self.args.seq_len, :]
                original_seq_pred = original_seq[:, self.args.seq_len:, :]
                rec_history = rec[:, :self.args.seq_len, :]
                rec_pred = rec[:, self.args.seq_len:, :]
                history_time_score = torch.mean(self.temp_anomaly_criterion(original_seq_history, rec_history),
                                                dim=-1)
                pred_time_score = torch.mean(self.temp_anomaly_criterion(original_seq_pred, rec_pred), dim=-1)
                time_score = 0 * history_time_score + 1 * pred_time_score
                history_freq_score = torch.mean(self.freq_anomaly_criterion(original_seq_history, rec_history),
                                                dim=-1)
                pred_freq_score = torch.mean(self.freq_anomaly_criterion(original_seq_pred, rec_pred), dim=-1)
                freq_score = 0 * history_freq_score + 1 * pred_freq_score
                score = (time_score + self.args.score_lambda * freq_score).detach().cpu().numpy()

                attens_energy.append(score)
                real_labels.append(batch_y_label)
                preds_series.append(pred.detach().cpu().numpy())
                preds_series_rec.append(rec.detach().cpu().numpy())
                true_series.append(batch_y.detach().cpu().numpy())
        attens_energy = np.concatenate(attens_energy, axis=0)
        thre_energy = np.array(attens_energy)
        thre_win_score = max_per_row(thre_energy)
        print("---------The forward propagation of the test set has been completed---------")

        if not isinstance(self.args.ratio, list):
            self.args.ratio = [self.args.ratio]

        preds = {}
        for ratio in self.args.ratio:
            threshold = np.percentile(train_energy, 100 - ratio)
            preds[ratio] = recon_labels(thre_energy, threshold)

        real_labels = np.concatenate(real_labels, axis=0)
        real_labels = np.array(real_labels).astype(int)
        real_labels = recon_labels(real_labels, 0.5)
        plot_label(pred_label=preds[3], true_label=real_labels, show_len=10000, save_path=result_path, save_name='label_plot.png')

        accuracy_list = []
        precision_list = []
        recall_list = []
        F_score_list = []

        for ratio, pred in preds.items():
            self.args.ratio = ratio

            print("When the threshold is {}%, the evaluation indicators are as follows: ".format(100 - ratio))

            print("pred: ", pred.shape)
            print("gt:", real_labels.shape)

            result_str = write_metrics(actual=real_labels, predicted=pred, args=self.args,
                          file_path=result_path + "result_anomaly_prediction.txt")

            print("When the threshold is {}%, all the evaluations have been calculated.".format(100-ratio))

            accuracy_line = [line for line in result_str.split("\n") if "Accuracy" in line][0]
            accuracy_value = float(accuracy_line.split(" : ")[1])
            accuracy_list.append(accuracy_value)

            precision_line = [line for line in result_str.split("\n") if "Precision" in line][0]
            precision_value = float(precision_line.split(" : ")[1])
            precision_list.append(precision_value)

            recall_line = [line for line in result_str.split("\n") if "Recall" in line][0]
            recall_value = float(recall_line.split(" : ")[1])
            recall_list.append(recall_value)

            F_score_line = [line for line in result_str.split("\n") if "F-score" in line][0]
            F_score_value = float(F_score_line.split(" : ")[1])
            F_score_list.append(F_score_value)

        find_max_fscore(preds, accuracy_list, precision_list, recall_list, F_score_list, save_path=result_path, save_name='best_result.txt')
        return
