import torch
import numpy as np


def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def t2v_metrics(sims, query_masks=None):
    """Compute retrieval metrics from a similiarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing queries from the dataset (two videos
             in MSRVTT only have 19, rather than 20 captions)

    Returns:
        (dict[str:float]): retrieval metrics
    """
    assert sims.ndim == 2, "expected a matrix"
    num_queries, num_vids = sims.shape
    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    if False:
        import sys
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
        from zsvision.zs_iterm import zs_dispFig # NOQA
        plt.matshow(dists)
        zs_dispFig()

    # The indices are computed such that they slice out the ground truth distances
    # from the psuedo-rectangular dist matrix
    queries_per_video = num_queries // num_vids
    gt_idx = [[np.ravel_multi_index([ii, jj], (num_queries, num_vids))
              for ii in range(jj * queries_per_video, (jj + 1) * queries_per_video)]
              for jj in range(num_vids)]
    gt_idx = np.array(gt_idx)
    gt_dists = dists.reshape(-1)[gt_idx.reshape(-1)]
    gt_dists = gt_dists[:, np.newaxis]
    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    # note: we sometimes need to break ties (these should occur extremely rarely)
    if rows.size > num_queries:
        _, idx = np.unique(rows, return_index=True)
        cols = cols[idx]
    msg = "expected ranks to match queries ({} vs {}) "
    if cols.size != num_queries:
        import ipdb; ipdb.set_trace()
    assert cols.size == num_queries, msg

    if False:
        # overload mask to check that we can recover the scores for single-query
        # retrieval
        print("DEBUGGING MODE")
        query_masks = np.zeros_like(query_masks)
        query_masks[:, 0] = 1  # recover single query score

    if query_masks is not None:
        # remove invalid queries
        assert query_masks.size == num_queries, "invalid query mask shape"
        cols = cols[query_masks.reshape(-1).astype(np.bool)]
        assert cols.size == query_masks.sum(), "masking was not applied correctly"
        # update number of queries to account for those that were missing
        num_queries = query_masks.sum()

    if False:
        # sanity check against old logic for square matrices
        gt_dists_old = np.diag(dists)
        gt_dists_old = gt_dists_old[:, np.newaxis]
        _, cols_old = np.where((sorted_dists - gt_dists_old) == 0)
        assert np.array_equal(cols_old, cols), "new metric doesn't match"

    return cols2metrics(cols, num_queries)


def v2t_metrics(sims, query_masks=None):
    """Compute retrieval metrics from a similiarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing captions from the dataset

    Returns:
        (dict[str:float]): retrieval metrics

    NOTES: We find the closest "GT caption" in the style of VSE, which corresponds
    to finding the rank of the closest relevant caption in embedding space:
    github.com/ryankiros/visual-semantic-embedding/blob/master/evaluation.py#L52-L56
    """
    # switch axes of text and video
    sims = sims.T

    if False:
        # experiment with toy example
        sims = np.ones((3, 3))
        sims[0, 0] = 2
        sims[1, 1:2] = 2
        sims[2, :] = 2
        query_masks = None

    assert sims.ndim == 2, "expected a matrix"
    num_queries, num_caps = sims.shape
    dists = -sims
    caps_per_video = num_caps // num_queries

    MISSING_VAL = 1E8
    query_ranks = []
    for ii in range(num_queries):
        row_dists = dists[ii, :]
        if query_masks is not None:
            # Set missing queries to have a distance of infinity.  A missing query
            # refers to a query position `n` for a video that had less than `n`
            # captions (for example, a few MSRVTT videos only have 19 queries)
            row_dists[np.logical_not(query_masks.reshape(-1))] = MISSING_VAL

        # NOTE: Using distance subtraction to perform the ranking is easier to make
        # deterministic than using argsort, which suffers from the issue of defining
        # "stability" for equal distances.  Example of distance subtraction code:
        # github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/train.py
        sorted_dists = np.sort(row_dists)

        min_rank = np.inf
        for jj in range(ii * caps_per_video, (ii + 1) * caps_per_video):
            if row_dists[jj] == MISSING_VAL:
                # skip rankings of missing captions
                continue
            rank = np.where((sorted_dists - row_dists[jj]) == 0)[0][0]
            if rank < min_rank:
                min_rank = rank
        query_ranks.append(min_rank)
    query_ranks = np.array(query_ranks)

    # sanity check against old version of code
    if False:
        sorted_dists = np.sort(dists, axis=1)
        gt_dists_old = np.diag(dists)
        gt_dists_old = gt_dists_old[:, np.newaxis]
        rows_old, cols_old = np.where((sorted_dists - gt_dists_old) == 0)
        if rows_old.size > num_queries:
            _, idx = np.unique(rows_old, return_index=True)
            cols_old = cols_old[idx]
        num_diffs = (1 - (cols_old == query_ranks)).sum()
        msg = f"new metric doesn't match in {num_diffs} places"
        assert np.array_equal(cols_old, query_ranks), msg

        # visualise the distance matrix
        import sys
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
        from zsvision.zs_iterm import zs_dispFig # NOQA
        plt.matshow(dists)
        zs_dispFig()

    return cols2metrics(query_ranks, num_queries)


def cols2metrics(cols, num_queries):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
    metrics["R5"] = 100 * float(np.sum(cols < 5)) / num_queries
    metrics["R10"] = 100 * float(np.sum(cols < 10)) / num_queries
    metrics["R50"] = 100 * float(np.sum(cols < 50)) / num_queries
    metrics["MedR"] = np.median(cols) + 1
    metrics["MeanR"] = np.mean(cols) + 1
    return metrics

