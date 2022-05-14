import numpy as np

import util


def replace_vip_with_halftoning_directly(img_in, img_encrypt):
    img_halftoning = color_halftoning(img_in)
    mask_vip = (img_encrypt < 0)
    img_result = img_encrypt.copy() * 255
    img_result[mask_vip] = img_halftoning[mask_vip]

    assert np.all(img_result < 256)
    assert np.all(img_result >= 0)
    img_result = img_result.astype(np.uint8)
    util.check_binary(img_result, bmax=255)
    return img_result


def error_diffusion(img_in, img_encrypt, method='Floyd_Steinberg',
                    threshold_modulation=False,
                    scale_factor=None, clip_factor=None):
    """Color Extended Visual Cryptography Part C

    method(str): 'Floyd_Steinberg' or 'Jarvis'
    """
    if method == 'Floyd_Steinberg':
        diffusion_mask = np.array([
            [0, 0, 7],
            [3, 5, 1]
        ]) / 16
    elif method == 'Jarvis':
        diffusion_mask = np.array([
            [0, 0, 0, 7, 5],
            [3, 5, 7, 5, 3],
            [1, 3, 5, 3, 1],
        ]) / 48
    else:
        raise ValueError("method should be 'Floyd_Steinberg' or 'Jarvis'")

    img_result = error_diffusion_by_mask(
        img_in, img_encrypt, diffusion_mask,
        threshold_modulation,
        scale_factor=scale_factor, clip_factor=clip_factor
    )
    return img_result


def error_diffusion_by_mask(img_in, img_encrypt, diffusion_mask,
                            threshold_modulation=False,
                            scale_factor=None,
                            clip_factor=None):
    assert img_in.shape == img_encrypt.shape
    assert len(img_in.shape) == 3
    assert diffusion_mask.shape[1] % 2 == 1

    mask_encrypt = (img_encrypt >= 0)
    assert not np.any(mask_encrypt[:, :, 0] ^ mask_encrypt[:, :, 1])
    assert not np.any(mask_encrypt[:, :, 0] ^ mask_encrypt[:, :, 2])
    mask_fixed = mask_encrypt[:, :, 0]  # (row, col)

    pad_row = diffusion_mask.shape[0] - 1
    pad_col = diffusion_mask.shape[1] // 2

    img_working_pad = np.pad(
        img_in, [(0, pad_row), (pad_col, pad_col), (0, 0)],
        'constant', constant_values=0)
    img_working_pad = img_working_pad.astype(float) / 255

    img_encrypt_pad = np.pad(
        img_encrypt, [(0, pad_row), (pad_col, pad_col), (0, 0)],
        'constant', constant_values=0)
    mask_fixed_pad = np.pad(
        mask_fixed, [(0, pad_row), (pad_col, pad_col)],
        'constant', constant_values=0)

    for i_row in range(img_in.shape[0]):
        for i_col in range(pad_col, img_in.shape[1] + pad_col):
            is_fixed = mask_fixed_pad[i_row, i_col]
            for i_channel in range(3):
                pos_point = (i_row, i_col, i_channel)
                pos_diffusion = (slice(i_row, i_row + pad_row + 1),
                                 slice(i_col - pad_col,
                                       i_col + pad_col + 1),
                                 i_channel)

                old_value = img_working_pad[pos_point]
                # if img_working_pad[pos_point] > 0.5 + clip_factor:
                #     old_value = 0.5 + clip_factor
                # elif img_working_pad[pos_point] < 0.5 - clip_factor:
                #     old_value = 0.5 - clip_factor

                if is_fixed:
                    new_value = img_encrypt_pad[pos_point]
                else:
                    thresh = _calc_threshold(
                        img_working_pad[i_row, :, i_channel],
                        threshold_modulation, pos_point
                    )
                    new_value = 1 if old_value > thresh else 0

                img_working_pad[pos_point] = new_value

                error_value = old_value - new_value
                if scale_factor:
                    error_value *= scale_factor
                if clip_factor:
                    if error_value > clip_factor:
                        error_value = clip_factor
                    elif error_value < -clip_factor:
                        error_value = -clip_factor

                error_diffusion = error_value * diffusion_mask
                img_working_pad[pos_diffusion] += error_diffusion

    effective_range = (slice(None, img_in.shape[0]),
                       slice(pad_col, img_in.shape[1] + pad_col),
                       slice(None))
    img_out = img_working_pad[effective_range] * 255

    img_out = util.convert_uint8(img_out)
    util.check_binary(img_out, bmax=255)
    return img_out


def _calc_threshold(one_row, threshold_modulation: bool, position=None):
    _, i_col, _ = position

    if threshold_modulation:
        idx_start = max(i_col - 3, 0)
        thresh = 0.25 + 0.25 * np.mean(one_row[idx_start: i_col])
    else:
        thresh = 0.5
    return thresh


def color_halftoning(img_in, method='Floyd_Steinberg'):
    """Channelwise halftoning via error diffusion

    method should be 'Floyd_Steinberg' or 'Jarvis'
    """
    img_encrypt = -np.ones(img_in.shape, dtype=int)
    img_halftoning = error_diffusion(
        img_in, img_encrypt=img_encrypt, method=method)
    return img_halftoning


def error_filter_convolution(img_in, img_encrypt, method='Floyd_Steinberg', threshold_modulation=False):
    if method == 'Floyd_Steinberg':
        error_filter = np.array([
            [0, 0, 7],
            [3, 5, 1]
        ]) / 16
    elif method == 'Jarvis':
        error_filter = np.array([
            [0, 0, 0, 7, 5],
            [3, 5, 7, 5, 3],
            [1, 3, 5, 3, 1],
        ]) / 48
    else:
        raise ValueError("method should be 'Floyd_Steinberg' or 'Jarvis'")

    img_result = error_filter_convolution_given_filter(
        img_in, img_encrypt, error_filter, threshold_modulation
    )
    return img_result


def error_filter_convolution_given_filter(img_in, img_encrypt, error_filter, threshold_modulation=False):
    assert img_in.shape == img_encrypt.shape
    assert len(img_in.shape) == 3

    kernel = error_filter[::-1, ::-1]
    pad_row = kernel.shape[0] - 1
    pad_col = kernel.shape[1] // 2

    mask_encrypt = (img_encrypt >= 0)
    assert not np.any(mask_encrypt[:, :, 0] ^ mask_encrypt[:, :, 1])
    assert not np.any(mask_encrypt[:, :, 0] ^ mask_encrypt[:, :, 2])
    mask_fixed = mask_encrypt[:, :, 0]  # (row, col)

    img_origin_pad = np.pad(
        img_in, [(pad_row, 0), (pad_col, pad_col), (0, 0)],
        'constant', constant_values=0)
    img_origin_pad = img_origin_pad.astype(float) / 255
    img_result_pad = np.zeros(img_origin_pad.shape)

    img_encrypt_pad = np.pad(
        img_encrypt, [(pad_row, 0), (pad_col, pad_col), (0, 0)],
        'constant', constant_values=0)
    mask_fixed_pad = np.pad(
        mask_fixed, [(pad_row, 0), (pad_col, pad_col)],
        'constant', constant_values=0)

    for i_row in range(pad_row, img_in.shape[0] + pad_row):
        for i_col in range(pad_col, img_in.shape[1] + pad_col):
            is_fixed = mask_fixed_pad[i_row, i_col]
            for i_channel in range(3):
                pos_point = (i_row, i_col, i_channel)
                pos_diffusion = (slice(i_row - pad_row, i_row + 1),
                                 slice(i_col - pad_col, i_col + pad_col + 1),
                                 i_channel)
                diffusion_error = np.sum(
                    kernel * (img_origin_pad[pos_diffusion] - img_result_pad[pos_diffusion]))
                old_value = img_origin_pad[pos_point] + diffusion_error
                if is_fixed:
                    new_value = img_encrypt_pad[pos_point]
                else:
                    thresh = _calc_threshold(
                        img_result_pad[i_row, :, i_channel],
                        threshold_modulation, pos_point
                    )
                    new_value = 1 if old_value > thresh else 0
                img_result_pad[pos_point] = new_value

    effective_range = (slice(pad_row, img_in.shape[0] + pad_row),
                       slice(pad_col, img_in.shape[1] + pad_col),
                       slice(None))
    img_out = img_result_pad[effective_range] * 255

    img_out = util.convert_uint8(img_out)
    util.check_binary(img_out, bmax=255)
    return img_out


def multiscale_error_diffusion(img_in, encryption_share):
    img_out = np.zeros(img_in.shape, dtype=np.uint8)
    for idx in range(3):
        img_out[:, :, idx] = MultiscaleErrorDiffusionGray(
            img_in[:, :, idx], encryption_share[:, :, idx])()
    return img_out


class MultiscaleErrorDiffusionGray:
    def __init__(self, img_in, encryption_share) -> None:
        assert img_in.shape[0] == img_in.shape[1]
        assert encryption_share.ndim == 2

        self.img_origin_normalized = img_in.astype(float) / 255
        self.img_shape = img_in.shape
        self.img_out = (encryption_share >= 0).astype(int)
        self.error_pyramid = self._create_pyramid()
        self.num_layer = len(self.error_pyramid)
        self.is_fixed = (encryption_share >= 0)

    def __call__(self, threshold=0.5):
        """Only implement for power of 2"""
        while self.error_pyramid[0] >= threshold:
            pos_max = self._maximum_intensity_guadiance()
            if not self.is_fixed[pos_max]:
                self.img_out[pos_max] = 255
            self._bottom_layer_diffusion(pos_max)

        self.img_out = util.convert_uint8(self.img_out)
        util.check_binary(self.img_out, bmax=255)
        return self.img_out

    def _create_pyramid(self):
        lg = self.img_shape[0]
        pyramid = [self.img_origin_normalized - self.img_out]
        while lg > 1:
            idx = self._get_idx_vectorized((lg, lg), kernel_size=2, stride=2)
            shape_half = (lg >> 1, lg >> 1)
            pyramid.append(
                np.sum(pyramid[-1].ravel()[idx], axis=-1).reshape(shape_half))
            lg >>= 1
        return pyramid[::-1]

    def _get_idx_vectorized(self, shape_in, kernel_size=2, stride=None):
        if not stride:
            stride = kernel_size

        idx_base = (np.expand_dims(np.arange(kernel_size), 0) +
                    np.expand_dims(np.arange(kernel_size) * shape_in[1], 1))

        expand_row = np.arange(
            shape_in[1] - kernel_size + 1, step=stride)
        idx_row = (np.expand_dims(idx_base.ravel(), 0) +
                   np.expand_dims(expand_row, -1))

        expand_col = (
            shape_in[1] * np.arange(shape_in[0] - kernel_size + 1, step=stride))
        idx_all = (
            np.expand_dims(idx_row.ravel(), 0) + np.expand_dims(expand_col, 1))

        reshape_size = len(idx_base.ravel())
        idx_all = idx_all.reshape(-1, reshape_size)
        return idx_all

    def _maximum_intensity_guadiance(self):
        position = (0, 0)
        for layer_number in range(1, len(self.error_pyramid)):
            position = self._find_maximum_to_bottom_layer(
                position, layer_number)
        return position

    def _find_maximum_to_bottom_layer(self, pos, next_layer_number):
        pos_next = (pos[0] * 2, pos[1] * 2)
        idx_row = [pos_next[0], pos_next[0], pos_next[0] + 1, pos_next[0] + 1]
        idx_col = [pos_next[1], pos_next[1] + 1, pos_next[1], pos_next[1] + 1]
        idx_max = np.argmax(
            self.error_pyramid[next_layer_number][idx_row, idx_col])
        if idx_max > 4:
            print(next_layer_number)
            print(idx_row)
            print(idx_max)
            print(self.error_pyramid[next_layer_number][idx_row, idx_col])
        return idx_row[idx_max], idx_col[idx_max]

    def _bottom_layer_diffusion(self, position) -> None:
        b_top, b_bottom = 0, self.img_shape[0] - 1
        b_left, b_right = 0, self.img_shape[1] - 1

        row_range = list(range(max(position[0] - 1, b_top),
                               min(position[0] + 2, b_bottom + 1)))
        col_range = list(range(max(position[1] - 1, b_left),
                               min(position[1] + 2, b_right + 1)))
        idx_row = []
        for rv in row_range:
            idx_row.extend([rv] * len(col_range))
        idx_col = col_range * len(row_range)

        filter_corner = np.array([[0, 2],
                                  [2, 1]]) / 5
        filter_side = np.array([[2, 0, 2],
                                [1, 2, 1]]) / 8
        filter_middle = np.array([[1, 2, 1],
                                  [2, 0, 2],
                                  [1, 2, 1]]) / 12
        if position == (b_top, b_left):
            diffusion_filter = filter_corner.copy()
        elif position == (b_top, b_right):
            diffusion_filter = filter_corner[:, ::-1]
        elif position == (b_bottom, b_left):
            diffusion_filter = filter_corner[::-1, :]
        elif position == (b_bottom, b_right):
            diffusion_filter = filter_corner[::-1, ::-1]
        elif position[0] == b_top:
            diffusion_filter = filter_side.copy()
        elif position[0] == b_bottom:
            diffusion_filter = filter_side[::-1, :]
        elif position[1] == b_left:
            diffusion_filter = filter_side.T
        elif position[1] == b_right:
            diffusion_filter = filter_side.T[:, ::-1]
        else:
            diffusion_filter = filter_middle.copy()

        if self.is_fixed[position]:
            error_to_diffuse = (
                self.error_pyramid[-1][position] - self.img_out[position])
        else:
            error_to_diffuse = self.error_pyramid[-1][position] - 1
            # self.is_fixed[position] = True
        err_dfsn = error_to_diffuse * diffusion_filter.ravel()
        self.error_pyramid[-1][idx_row, idx_col] += err_dfsn

        self._update_error_pyramid(-self.error_pyramid[-1][position], position)
        for err, ir, ic in zip(err_dfsn, idx_row, idx_col):
            self._update_error_pyramid(err, (ir, ic))

        self.error_pyramid[-1][position] = 0

    def _update_error_pyramid(self, err, pos_in):
        pos_curr = pos_in
        for layer_number in range(self.num_layer - 2, -1, -1):
            pos_prev = (pos_curr[0] >> 1, pos_curr[1] >> 1)
            self.error_pyramid[layer_number][pos_prev] += err
            pos_curr = pos_prev
