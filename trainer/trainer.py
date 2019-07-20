import numpy as np
import torch
from contextlib import contextmanager
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loaders,
                 lr_scheduler, visualizer, disable_nan_checks, mini_train=False):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loaders = data_loaders
        self.lr_scheduler = lr_scheduler
        self.mini_train = mini_train
        self.disable_nan_checks = disable_nan_checks
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
                prog = self._progress(batch_idx)
                self.logger.debug(f"Train Epoch: {epoch} {prog} Loss: {loss.item():.6f}")

            if batch_idx == self.len_epoch or (self.mini_train and batch_idx > 3):
                break

        log = {'loss': total_loss / self.len_epoch}
        val_log = self._valid_epoch(epoch)
        log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def verbose(self, epoch, metrics, mode, name="TEST"):
        r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
        msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
        msg += f", R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
        msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
        print(msg)
        # print(msg.format(mode, name, epoch, 100 * r1, 100 * r5, 100 * r10, 100 * r50,
        #                  mdr, mnr))

    def log_metrics(self, metric_store, epoch, metric_name):
        for key, value in metric_store.items():
            self.writer.add_scalar(f"{metric_name}/{key}", value, epoch)

    @contextmanager
    def valid_samples(self, retrieval_samples):
        """Provide a context for managing temporary, cloned copies of retrieval
        sample tensors.

        The rationale here is that to use nan-checking in the model (to validate the
        positions of missing experts), we need to modify the underlying tensors. This
        function lets the evaluation code run (and modify) temporary copies, without
        modifying the originals.
        """
        if self.disable_nan_checks:
            print("running without nan checks")
            yield retrieval_samples
        else:
            exp_dict = retrieval_samples["experts"].items()
            experts = {key: val.clone().to(self.device) for key, val in exp_dict}
            retrieval_samples_ = {
                "experts": experts,
                "ind": retrieval_samples["ind"],
                "text": retrieval_samples["text"].to(self.device),
            }
            try:
                yield retrieval_samples_
            finally:
                del retrieval_samples_

    def _valid_epoch(self, epoch):
        """Validate model after an epoch of training and store results to disk.

        Args:
            epoch (int): the current epoch

        Returns:
            A log that contains information about validation

        NOTE: The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        self.writer.mode = "val"
        with torch.no_grad():
            retrieval_samples, meta = self.data_loaders["retrieval"]

            # To use the nan-checks safely, we need make temporary copies of the data
            with self.valid_samples(retrieval_samples) as samples:
                output = self.model(**samples)

            sims = output["cross_view_conf_matrix"].data.cpu().float().numpy()
            dataset = self.data_loaders.dataset_name
            nested_metrics = {}
            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(sims, query_masks=meta["query_masks"])
                self.verbose(epoch=epoch, metrics=res, name=dataset, mode=metric_name)
                self.log_metrics(metric_store=res, epoch=epoch, metric_name=metric_name)
                nested_metrics[metric_name] = res

        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, bins='auto')
        if self.data_loaders.num_test_captions == 1:
            self.visualizer.visualize_ranking(
                sims=sims,
                meta=meta,
                epoch=epoch,
                nested_metrics=nested_metrics,
            )
        return {"nested_val_metrics": nested_metrics}

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loaders, 'n_samples'):
            current = batch_idx * self.data_loaders.batch_size
            total = self.data_loaders.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
