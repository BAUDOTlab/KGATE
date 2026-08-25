from torch import ones_like, tensor
from torch.nn import Module, MarginRankingLoss, BCELoss

class Loss(Module):
    def __init__(self, *terms: list[Module]):
        self.terms = terms

    def forward(self, positive_scores: tensor, negative_scores: tensor):
        return sum([term(positive_scores, negative_scores) for term in self.terms])

    def add_term(self, new_term: Module):
        self.terms.append(new_term)
