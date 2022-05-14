import numpy as np

from basis_generation import BasisMatrices
import util


def distribute_matrices(img_message,
                        basis_matrices: BasisMatrices,
                        output_shape):
    assert len(output_shape) == 3
    assert output_shape[-1] == 3

    num_share = basis_matrices.shape[1]
    expand_row = basis_matrices.expand_row
    expand_col = basis_matrices.expand_col

    assert output_shape[0] == img_message.shape[0] * expand_row
    assert output_shape[1] == img_message.shape[1] * expand_col
    assert basis_matrices.shape[-1] == expand_row * expand_col

    # shares, row, col, channel, pattern_size
    img_share = np.zeros((num_share, *output_shape), dtype=np.int)

    util.check_binary(img_message, bmax=255)
    binary_message = (img_message > 0).astype(int)

    for i_row in range(img_message.shape[0]):
        for i_col in range(img_message.shape[1]):
            basis_permutated = basis_matrices.column_permutation()
            row_start = i_row * expand_row
            col_start = i_col * expand_col
            for i_channel in range(3):
                pos_message = (i_row, i_col, i_channel)
                pos_share = (slice(row_start, row_start + expand_row),
                             slice(col_start, col_start + expand_col),
                             i_channel)
                for i_share in range(num_share):
                    message_bit = binary_message[pos_message]
                    subpixel = basis_permutated[message_bit, i_share]
                    subpixel = subpixel.reshape(expand_row, expand_col)
                    img_share[(i_share, *pos_share)] = subpixel
    return img_share
