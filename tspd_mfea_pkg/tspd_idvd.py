from typing import Tuple, Any

from mfea_pkg.individual import Individual


class TSPD_Individual(Individual):
    def __init__(self, genes: Tuple[Any, Any], num_task):
        super().__init__(genes, num_task)
        self.genes_len = len(genes[0])

    def __repr__(self):
        ret = []
        order, color = self.genes
        for i in range(self.genes_len):
            if color[i] == " ":
                ret.append(f"{order[i]}")
            else:
                ret.append(f"{order[i]}{color[i]}")
        return"[" + " ".join(ret) + "]"
