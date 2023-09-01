import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
from .. import util
from . import gBCELoss

class MultipleNegativesRankingLossGanesh(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.

        For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
        n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly.

        The performance usually increases with increasing batch sizes.

        For more information, see: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)

        Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.

        Example::

            from sentence_transformers import SentenceTransformer, losses, InputExample
            from torch.utils.data import DataLoader

            model = SentenceTransformer('distilbert-base-uncased')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, corpus,  t: float = 0.75,similarity_fct=util.cos_sim):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(MultipleNegativesRankingLossGanesh, self).__init__()
        self.model = model
        # self.scale = scale
        self.similarity_fct = similarity_fct
        self.t = t
        self.loss = gBCELoss.gBCE
        self.corpus= corpus


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        scores = torch.mm(embeddings_a, embeddings_b.transpose(0,1))/100-2 
        print('scores tensors',scores)
        alpha=(scores.shape[1]/(len(self.corpus)-1))
        beta=alpha*((self.t*(1-1/alpha) + 1/alpha))
        labels_raw = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  
        labels = torch.nn.functional.one_hot(labels_raw, num_classes=scores.shape[1])
        mean_loss= torch.mean(self.loss(scores, labels, beta))
        print("mean loss", mean_loss)
        return mean_loss

    def get_config_dict(self):
        return { 'similarity_fct': self.similarity_fct.__name__}





