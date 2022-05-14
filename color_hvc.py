import os
import argparse

import numpy as np
import cv2

import basis_generation
import error_diffusion
import matrices_distribution


class ColorHVC:
    def __init__(self, vc_scheme=(2, 2), message_resolution=(128, 128)) -> None:
        if vc_scheme not in {(2, 2), (3, 4)}:
            raise ValueError(
                f"Only support (2, 2) or (3, 4) scheme, get {vc_scheme}")

        self.vc_scheme = vc_scheme
        self.num_shares = vc_scheme[1]

        self.message_resolution = message_resolution
        self.basis_matrices = basis_generation.BasisMatrices(
            vc_scheme=vc_scheme)
        self.share_resolution = self._get_share_resolution()

        self._cmy_inputs = None
        self._cmy_message = None
        self._encryption_shares = None

    def prepare(self, img_message):
        self._check_and_save_message_image(img_message)
        halftoning_message = error_diffusion.color_halftoning(
            self._cmy_message)
        self._encryption_shares = matrices_distribution.distribute_matrices(
            halftoning_message, self.basis_matrices, (*self.share_resolution, 3))

    def encrypt(self, img_inputs, img_message=None):
        if img_message is not None:
            self.prepare(img_message)
        self._check_and_save_input_image(img_inputs)

        cmy_shares = []
        for idx in range(self.num_shares):
            cmy_shares.append(error_diffusion.error_diffusion(
                self._cmy_inputs[idx], self._encryption_shares[idx],
                method='Floyd_Steinberg', threshold_modulation=True,
                scale_factor=self.basis_matrices.vip_ratio))
        rgb_shares = [conver_rgb_cmy(img) for img in cmy_shares]
        return rgb_shares

    def _check_and_save_message_image(self, img_message):
        if img_message.shape[:2] != self.message_resolution:
            img_resize = cv2.resize(
                img_message, self.message_resolution[::-1])
            self._cmy_message = conver_rgb_cmy(img_resize)
        else:
            self._cmy_message = conver_rgb_cmy(img_message)

    def _check_and_save_input_image(self, img_inputs):
        if len(img_inputs) != self.vc_scheme[1]:
            raise ValueError(
                f"number of img_inputs should be {self.vc_scheme[1]}")

        self._cmy_inputs = []
        for image in img_inputs:
            if image.shape[:2] != self.share_resolution:
                img_resize = cv2.resize(image, self.share_resolution[::-1])
                self._cmy_inputs.append(conver_rgb_cmy(img_resize))
            else:
                self._cmy_inputs.append(conver_rgb_cmy(image))

    def _get_share_resolution(self):
        expand_row = self.basis_matrices.expand_row
        expand_col = self.basis_matrices.expand_col
        share_row = self.message_resolution[0] * expand_row
        share_col = self.message_resolution[1] * expand_col
        return share_row, share_col


def conver_rgb_cmy(img_in):
    return 255 - img_in


def decrypt(img_shares):
    cmy_recover = np.zeros(img_shares[0].shape, dtype=bool)
    for image in img_shares:
        cmy_recover |= (conver_rgb_cmy(image) // 255).astype(bool)
    cmy_recover = cmy_recover.astype(np.uint8) * 255
    return conver_rgb_cmy(cmy_recover)


def get_args():
    parser = argparse.ArgumentParser(
        description='Color extended visual cryptography using error diffusion')
    parser.add_argument("-s", "--scheme", type=int, nargs=2,
                        help="VC scheme, (2, 2) or (3, 4)")
    parser.add_argument("-ip", "--input_path", type=str,
                        help="input images path", required=True)
    parser.add_argument("-if", "--input_fnames", type=str, nargs='+',
                        help="input images filenames", required=True)
    parser.add_argument("-m", "--message", type=str,
                        help="message image filename")
    parser.add_argument("-op", "--output_path", type=str,
                        help="output images path", required=True)
    parser.add_argument("-oef", "--output_fnames", type=str, nargs='+',
                        help="encrypted share images filenames")
    parser.add_argument("-r", "--resolution", type=int, nargs=2,
                        help="destination message resolution to be resized", default=(128, 128))

    parser.add_argument("-d", "--decrypt", action="store_true")
    parser.add_argument("-odf", "--output_decrypt_fname", type=str,
                        help="recovered message filename")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    input_path = args.input_path
    output_path = args.output_path
    input_fnames = args.input_fnames
    img_inputs = [cv2.imread(os.path.join(input_path, ifn))
                  for ifn in input_fnames]
    img_shares = None

    if args.message:
        assert len(args.resolution) == 2
        assert len(args.scheme) == 2

        vc_scheme = tuple(args.scheme)
        num_output = args.scheme[-1]
        if args.output_fnames:
            output_fnames = args.output_fnames
        else:
            output_fnames = [f"shares_{idx}.png" for idx in range(num_output)]

        img_message = cv2.imread(os.path.join(input_path, args.message))
        color_hvc = ColorHVC(
            vc_scheme=vc_scheme, message_resolution=args.resolution)
        img_shares = color_hvc.encrypt(img_inputs, img_message=img_message)

        for idx in range(num_output):
            out_pathname = os.path.join(output_path, output_fnames[idx])
            write_status = cv2.imwrite(out_pathname, img_shares[idx])
            if not write_status:
                raise ValueError(f"cannot wirte to {out_pathname}")

    if args.decrypt:
        if args.output_decrypt_fname:
            recover_fname = args.output_decrypt_fname
        else:
            recover_fname = 'message_recover.png'

        if img_shares is None:
            img_shares = img_inputs.copy()

        img_recover = decrypt(img_shares)
        cv2.imwrite(os.path.join(output_path, recover_fname), img_recover)
