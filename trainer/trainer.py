import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loaders,
                 lr_scheduler, visualizer, mini_train=False):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loaders = data_loaders
        self.lr_scheduler = lr_scheduler
        self.mini_train = mini_train
        self.len_epoch = len(self.data_loaders["train"])
        self.log_step = int(np.sqrt(data_loaders["train"].batch_size))
        self.visualizer = visualizer

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = 0
        for batch_idx, minibatch in enumerate(self.data_loaders["train"]):
            for key, val in minibatch["experts"].items():
                minibatch["experts"][key] = val.to(self.device)
            minibatch["text"] = minibatch["text"].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(**minibatch)
            loss = self.loss(output["cross_view_conf_matrix"])
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch or self.mini_train:
                break

        log = {'loss': total_loss / self.len_epoch}
        val_log = self._valid_epoch(epoch)
        log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def verbose(self, epoch, metrics, mode, name="TEST"):
        msg = "[{}]{:s} epoch {}, R@1: {:.1f}, R@5: {:.1f}, R@10 {:.1f}, "
        msg += "R@50 {:.1f}, MedR: {:g}, MeanR: {:.1f}"
        r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
        mdr, mnr = metrics["MedR"], metrics["MeanR"]
        print(msg.format(mode, name, epoch, 100 * r1, 100 * r5, 100 * r10, 100 * r50,
                         mdr, mnr))

    def log_metrics(self, metric_store, epoch, subset):
        for key, value in metric_store.items():
            self.writer.add_scalar("{}/{}".format(subset, key), value, epoch)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        # total_val_loss = 0
        # total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            retrieval_samples, meta = self.data_loaders["retrieval"]
            # experts_ = {}
            # for key, val in retrieval_samples["experts"].items():
            #     experts_[key] = val.clone().to(self.device)
            # retrieval_samples["text"] = 

            # To use the nan-checks safely, we need to copy the data
            retrieval_samples_ = {
                "text": retrieval_samples["text"].to(self.device),
                "experts": {key: val.clone().to(self.device)
                            for key, val in retrieval_samples["experts"].items()},
                "ind": retrieval_samples["ind"],
            }
            output = self.model(**retrieval_samples_)
            # loss = self.loss(output, target)

            # self.writer.add_scalar('loss', loss.item())
            # total_val_loss += loss.item()
            mat = output["cross_view_conf_matrix"].data.cpu().float().numpy()
            dataset = self.data_loaders.dataset_name
            metric_groups = {}
            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(mat, query_masks=meta["query_masks"])
                self.verbose(epoch=epoch, metrics=res, name=dataset, mode=metric_name)
                self.log_metrics(metric_store=res, epoch=epoch, subset="val")
                metric_groups[metric_name] = res

            del retrieval_samples_

            # total_val_metrics = self._eval_metrics(output, target)
            # self.writer.add_image('input', make_grid(data.cpu(), nrow=8,
            #                         normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        vis_vid_ranking = 5
        if (vis_vid_ranking and epoch % vis_vid_ranking == 0 and self.data_loaders.num_test_captions == 1):

            dists = -mat
            np.random.seed(0)
            sorted_ranks = np.argsort(dists, axis=1)
            gt_dists = np.diag(dists)
            rankings = []
            vis_top_k = 5
            hide_gt = False
            vis_rank_samples = 40
            # num_indep_samples = 1
            # random_seeds = np.arange(num_indep_samples)
            sample = np.random.choice(np.arange(dists.shape[0]), size=vis_rank_samples)
            for ii in sample:
                ranked_idx = sorted_ranks[ii][:vis_top_k]
                gt_captions = meta["raw_captions"][ii]
                # if args.sample_single_gt_caption:
                #     gt_captions = np.random.choice(gt_captions, 1).tolist()

                datum = {
                    "gt-sim": -gt_dists[ii],
                    "gt-captions": gt_captions,
                    "gt-rank": np.where(sorted_ranks[ii] == ii)[0][0],
                    "gt-path": meta["paths"][ii],
                    "top-k-sims": -dists[ii][ranked_idx],
                    "top-k-paths": np.array(meta["paths"])[ranked_idx],
                    "hide-gt": hide_gt,
                }
                rankings.append(datum)
            self.visualizer.display_current_results(
                rankings,
                epoch=epoch,
                metrics=metric_groups["t2v_metrics"],
            )
        return {}
        #     'val_metrics': (total_val_metrics / len(valid_data_loader)).tolist()
        # }
        # 'val_loss': total_val_loss / len(valid_data_loader),

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loaders, 'n_samples'):
            current = batch_idx * self.data_loaders.batch_size
            total = self.data_loaders.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
