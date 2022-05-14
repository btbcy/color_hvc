from abc import ABC, abstractmethod
import numpy as np


class BasisMatrices:
    def __init__(self, vc_scheme=(2, 2)):
        if vc_scheme not in {(2, 2), (3, 4)}:
            raise ValueError(
                f"Only support (2, 2) or (3, 4) scheme, get {vc_scheme}")
        self.scheme = SchemeTable.get_scheme(vc_scheme)
        self._basis_matrices = self.scheme.get_basis_matrices()
        self.pattern_size = self._basis_matrices.shape[-1]
        self.vip_ratio = (np.sum(self._basis_matrices[0, 0] < 0)
                          / self.pattern_size)
        self.expand_row, self.expand_col = self.scheme.get_row_col_expansion()

    @ property
    def basis_matrices(self):
        """dimension: on/off, share_group, pattern"""
        return self._basis_matrices

    @ property
    def shape(self):
        return self._basis_matrices.shape

    def column_permutation(self):
        permutation_index = np.random.permutation(self.pattern_size)
        return self._basis_matrices[:, :, permutation_index]


class VCScheme(ABC):
    @abstractmethod
    def get_basis_matrices():
        pass

    def get_row_col_expansion():
        pass


class Scheme22(VCScheme):
    def get_basis_matrices():
        return np.array(
            [[[1, 1, 0, -1], [1, -2, 0, 1]],
             [[0, 1, 1, -1], [1, -2, 0, 1]]],
            dtype=int
        )

    def get_row_col_expansion():
        return (2, 2)


class Scheme34(VCScheme):
    def get_basis_matrices():
        return np.array(
            [[[0, -1, 1, 1, 1, -1], [0, -2, 1, 1, -2, 1],
             [0, 1, -3, -3, 1, 1], [0, 1, -4, 1, -4, 1]],

             [[0, -1, 1, 1, 1, -1], [1, -2, 1, 1, -2, 0],
              [1, 1, -3, -3, 0, 1], [0, 1, -4, 1, -4, 1]]],
            dtype=int
        )

    def get_row_col_expansion():
        return (2, 3)


class SchemeTable:
    scheme_table = {
        (2, 2): Scheme22,
        (3, 4): Scheme34
    }

    def get_scheme(vc_scheme) -> VCScheme:
        return SchemeTable.scheme_table[vc_scheme]
