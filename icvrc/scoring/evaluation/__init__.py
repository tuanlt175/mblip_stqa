from .bleu import Bleu
from .cider import Cider

import numpy as np
import json


def compute_scores(gts, gens):
    bleu = Bleu()
    cider = Cider()
    bleu_scores, _ = bleu.compute_score(gts, gens)
    avg_bleu_score = np.array(bleu_scores).mean()
    cider_score, _ = cider.compute_score(gts, gens)

    return {
        "BLEU": avg_bleu_score,
        "CIDEr": cider_score
    }
