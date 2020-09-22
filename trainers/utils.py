import torch


def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_mrr_ndcgs_for_ks(scores, labels, ks):
    metrics = {}
    device = labels.device
    scores = scores

    labels = labels
    batch = labels.size()[0]

    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):

        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)

        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()

        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()

        zero_check = torch.ones([batch,1]).to(device)- torch.clamp(hits.sum(1),0,1).unsqueeze(1)
        e_hits = torch.cat((zero_check,hits),dim = 1)
        idx = torch.arange(e_hits.shape[1], 0, -1).to(device)
        tmp2= torch.einsum("ab,b->ab", (e_hits, idx)).to(device)
        indices = torch.argmax(tmp2, 1, keepdim=True)

        rec_rank = 1/indices.float()
        rec_rank[rec_rank == float("Inf")] = 0
        metrics['MRR@%d' % k] = rec_rank.mean().cpu().item()

    return metrics
