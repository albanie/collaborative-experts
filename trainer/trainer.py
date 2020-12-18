from contextlib import contextmanager

import copy
import torch
import numpy as np

from base import BaseTrainer
from utils import memory_summary
from model.metric import APMeter, APMeterChallenge

def verbose(epoch, metrics, mode, name="TEST"):
    r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
    msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
    msg += f", R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
    msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
    print(msg)


@contextmanager
def ctxt_mgr(samples, device, disable_nan_checks):
    """Provide a context for managing temporary, cloned copies of retrieval
    sample tensors.

    The rationale here is that to use nan-checking in the model (to validate the
    positions of missing experts), we need to modify the underlying tensors. This
    function lets the evaluation code run (and modify) temporary copies, without
    modifying the originals.
    """
    if disable_nan_checks:
        print("running without nan checks")
        yield samples
    else:
        exp_dict = samples["experts"].items()
        experts = {key: val.clone().to(device) for key, val in exp_dict}
        samples_ = {
            "experts": experts,
            "ind": samples["ind"],
            "text": samples["text"].to(device),
        }
        if "text_token_mask" in samples:
            samples_["text_token_mask"] = samples["text_token_mask"].to(device)
        try:
            yield samples_
        finally:
            del samples_


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loaders,
                 lr_scheduler, visualizer, disable_nan_checks, skip_first_n_saves,
                 include_optim_in_ckpts, force_cpu_val, distil_loss, distil_params, cache_targets=set(),
                 num_keep_ckpts=3, mini_train=False, val_freq=1, skip_tboard=False):
        super().__init__(model, loss, metrics, optimizer, config, mini_train=mini_train,
                         skip_tboard=skip_tboard, num_keep_ckpts=num_keep_ckpts)
        self.config = config
        self.cache_targets = cache_targets
        self.data_loaders = data_loaders
        self.lr_scheduler = lr_scheduler
        self.mini_train = mini_train
        self.disable_nan_checks = disable_nan_checks
        self.len_epoch = len(self.data_loaders["train"])
        self.log_step = int(np.sqrt(data_loaders["train"].batch_size))
        self.visualizer = visualizer
        self.force_cpu_val = force_cpu_val
        self.val_freq = val_freq
        self.skip_first_n_saves = skip_first_n_saves
        self.include_optim_in_ckpts = include_optim_in_ckpts
        self.seen = {"train": 0, "val": 0}
        self.distil_loss = distil_loss
        self.distil_params = distil_params
        self.tt_loss = torch.nn.SmoothL1Loss(reduction="elementwise_mean")

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
        total_loss = 0
        self.model.train()
        memory_summary()

        for batch_idx, minibatch in enumerate(self.data_loaders["train"]):
            for key, val in minibatch["experts"].items():
                minibatch["experts"][key] = val.to(self.device)

            for key in {"text", "text_token_mask"}:
                if key in minibatch:
                    minibatch[key] = minibatch[key].to(self.device)

            if "labels" in minibatch:
                labels = minibatch.pop("labels").to(self.device)

            if "distil_video" in minibatch:
                distil = minibatch.pop("distil_video")
                distil_text = minibatch.pop("distil_text")

                with torch.no_grad():
                    new_sims = None
                    for t in distil:
                        t_sim = None

                        for new_mod in distil[t]:
                            distil_text[t][new_mod] = distil_text[t][new_mod].to(self.device)
                            distil[t][new_mod] = distil[t][new_mod].to(self.device)
                            tmp_sim = torch.matmul(distil_text[t][new_mod].view(-1, distil_text[t][new_mod].shape[-1]), distil[t][new_mod].t())

                            if t_sim is None:
                                t_sim = tmp_sim
                            else:
                                t_sim = t_sim + tmp_sim

                        if new_sims is None:
                            new_sims = t_sim
                        else:
                            new_sims = new_sims + t_sim

                    new_sims = new_sims / len(distil.keys())

            self.optimizer.zero_grad()
            output = self.model(**minibatch)
            if "retrieval" in self.data_loaders.dataloaders:
                loss = self.loss(output["cross_view_conf_matrix"])
            else:
                loss = self.loss(x=output["class_preds"], target=labels)

            if self.distil_loss:
                loss += self.tt_loss(output["cross_view_conf_matrix"], new_sims)
            loss.backward()
            self.optimizer.step()

            sample_key = list(minibatch["experts"].keys())[0]
            batch_size = minibatch["experts"][sample_key].shape[0]
            self.seen["train"] += batch_size

            if not self.skip_tboard:
                # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.set_step(self.seen["train"], mode="train")
                self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()

            if batch_idx % self.log_step == 0:
                prog = self._progress(batch_idx)
                self.logger.info(f"Train Epoch: {epoch} {prog} Loss: {loss.item():.6f}")

            if batch_idx == self.len_epoch or (self.mini_train and batch_idx > 3):
                break

        log = {'loss': total_loss / self.len_epoch}
        if epoch % self.val_freq == 0:
            nested_log, cached_preds = self._valid_epoch(epoch)
            log.update(nested_log)
        else:
            nested_log, cached_preds = {}, None
            self.logger.info(f"skipping val for epoch: {epoch}")

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.logger.info(f"LR {self.lr_scheduler.get_lr()}")
        return log, cached_preds

    def log_metrics(self, metric_store, metric_name, mode):
        if not self.skip_tboard:
            print(f"logging metrics: {metric_name}")
            self.writer.set_step(step=self.seen[mode], mode=mode)
            for key, value in metric_store.items():
                self.writer.add_scalar(f"{metric_name}/{key}", value)

    def _valid_epoch(self, epoch):
        """Validate model after an epoch of training and store results to disk.

        Args:
            epoch (int): the current epoch

        Returns:
            A log that contains information about validation

        NOTE: The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        if not self.skip_tboard:
            self.writer.mode = "val"
        cached_preds = {key: {"vid_name": [], "preds": [], "labels": []}
                        for key in self.cache_targets}

        with torch.no_grad():
            if "retrieval" in self.data_loaders.dataloaders:
                samples, meta = self.data_loaders["retrieval"]

                sample_key = list(samples["experts"].keys())[0]
                batch_size = samples["experts"][sample_key].shape[0]
                self.seen["val"] += batch_size
                
                num_queries = samples["text"].shape[0] * samples["text"].shape[1]
                safe_queries = 256
                if num_queries > safe_queries:
                    partitions = int(np.ceil(num_queries / safe_queries))
                    chunk_size = int(np.ceil(samples["text"].shape[0] / partitions))
                    texts = copy.deepcopy(samples["text"])
                    sim_chunks = []
                    for chunk_idx in range(partitions):
                        chunk_start = chunk_idx * chunk_size
                        chunk_stop = (chunk_idx + 1) * chunk_size
                        samples["text"] = texts[chunk_start:chunk_stop]
                        if samples['text'].shape[0] == 0:
                            continue
                        with ctxt_mgr(samples, self.device,
                                      self.disable_nan_checks) as xx:
                            output = self.model(**xx)
                        sims = output["cross_view_conf_matrix"].data
                        sim_chunks.append(sims)

                    samples["text"] = texts  # restore pointer to original tensor
                    del texts
                    sims = torch.cat(sim_chunks, dim=0).data.cpu().float().numpy()
                else:
                    with ctxt_mgr(samples, self.device, self.disable_nan_checks) as xx:
                        output = self.model(**xx)
                    self.model = self.model.to(self.device)
                    sims = output["cross_view_conf_matrix"].data.cpu().float().numpy()

                # sample the loss (using only the first query for each video)
                queries_per_vid = meta["query_masks"].shape[1]
                sims_ = torch.from_numpy(sims).view(-1, queries_per_vid, sims.shape[-1])
                loss = self.loss(sims_[:, 0, :].contiguous())
                if not self.skip_tboard:
                    self.writer.add_scalar('first-query-loss', loss.item())
                dataset = self.data_loaders.dataset_name
                nested_metrics = {}
                for metric in self.metrics:
                    metric_name = metric.__name__
                    res = metric(sims, query_masks=meta["query_masks"])
                    if metric_name == "mean_average_precision":
                        print(f"Epoch: {epoch}, mean AP: {res['mAP']}")
                    else:
                        verbose(epoch=epoch, metrics=res, name=dataset, mode=metric_name)
                    self.log_metrics(res, metric_name=metric_name, mode="val")
                    nested_metrics[metric_name] = res

                # TODO(Samuel) disabled visualisation for now, simple to add in later
                num_test_caps = self.data_loaders.num_test_captions
                if num_test_caps == 1 and meta["raw_captions"] is not None:
                    if self.visualizer is not None:
                        self.visualizer.visualize_ranking(
                            sims=sims,
                            meta=meta,
                            epoch=epoch,
                            nested_metrics=nested_metrics,
                        )
                return {"nested_val_metrics": nested_metrics}, cached_preds

            elif "val" in self.data_loaders.dataloaders:
                metrics = [x() for x in self.metrics]
                for batch_idx, minibatch in enumerate(self.data_loaders["val"]):
                    for key, val in minibatch["experts"].items():
                        minibatch["experts"][key] = val.to(self.device)
                    labels = minibatch.pop("labels").to(self.device)
                    vid_name = minibatch.pop("vid_name")
                    output = self.model(**minibatch)
                    if "val" in self.cache_targets:
                        cached_preds["val"]["vid_name"].append(vid_name)
                        cached_preds["val"]["preds"].append(output["class_preds"])

                    for metric in metrics:
                        metric.add(output=output["class_preds"], target=labels)
                    if batch_idx % self.log_step == 0:
                        prog = self._progress(batch_idx)
                        self.logger.info(f"Val Epoch: {epoch} {prog}")
                
                nested_metrics = {}
                for metric in metrics:
                    if hasattr(metric, "topk"):
                        res = {f"top{key}": val for key, val in
                               zip(metric.topk, metric.value())}
                        self.log_metrics(res, mode="val", metric_name="accuracy")
                        nested_metrics["accuracy"] = res
                    elif isinstance(metric, APMeter):
                        res = {"mAP": metric.value().mean()}
                        self.log_metrics(res, mode="val",
                                         metric_name="mean_ap_non_challenge")
                        nested_metrics["mean_ap_non_challenge"] = res
                    elif isinstance(metric, APMeterChallenge):
                        res = {"mAP": metric.value().mean()}
                        self.log_metrics(res, mode="val",
                                         metric_name="mean_average_precision")
                        nested_metrics["mean_ap"] = res
                    else:
                        raise ValueError(f"unsupported mettric: {type(metric)}")
                nested = {"nested_val_metrics": nested_metrics}

                for target in self.cache_targets - {"val"}:
                    for batch_idx, minibatch in enumerate(self.data_loaders["tiny"]):
                        for key, val in minibatch["experts"].items():
                            minibatch["experts"][key] = val.to(self.device)
                        if "labels" in minibatch:
                            cached_preds[target]["labels"].append(minibatch.pop("labels"))
                        cached_preds[target]["vid_name"].append(minibatch.pop("vid_name"))
                        output = self.model(**minibatch)
                        cached_preds[target]["preds"].append(output["class_preds"])

                # aggregate all cached predictions
                for target in self.cache_targets:
                    for key, val in cached_preds[target].items():
                        cached_preds[key] = torch.cat(val).cpu().numpy()
                return nested, cached_preds

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loaders, 'n_samples'):
            current = batch_idx * self.data_loaders.batch_size
            total = self.data_loaders.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
