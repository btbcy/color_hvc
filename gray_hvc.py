import os
import argparse

import numpy as np
import cv2

import color_hvc


class GrayHVC:
    def __init__(self, vc_scheme=(2, 2), message_resolution=(128, 128)) -> None:
        self.color_hvc = color_hvc.ColorHVC(
            vc_scheme=vc_scheme, message_resolution=message_resolution)

    def prepare(self, img_message):
        self.color_hvc.prepare(repeat_grayscale_image(img_message))

    def encrypt(self, img_inputs, img_message=None):
        if img_message is not None:
            self.prepare(img_message)

        kkk_inputs = [repeat_grayscale_image(img) for img in img_inputs]
        rgb_shares = self.color_hvc.encrypt(img_inputs=kkk_inputs)
        gray_shares = [img[:, :, 0] for img in rgb_shares]
        return gray_shares


def repeat_grayscale_image(img_gray):
    if img_gray.ndim == 2:
        kkk_message = np.repeat(img_gray[:, :, np.newaxis], 3, axis=2)
    else:
        kkk_message = img_gray.copy()
    return kkk_message


def decrypt(img_shares):
    return color_hvc.decrypt(img_shares)


def get_args():
    parser = argparse.ArgumentParser(
        description='Grayscale wrapper for color extended visual cryptography using error diffusion')
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
    img_inputs = [cv2.imread(os.path.join(input_path, ifn), cv2.IMREAD_GRAYSCALE)
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

        img_message = cv2.imread(os.path.join(
            input_path, args.message), cv2.IMREAD_GRAYSCALE)
        gray_hvc = GrayHVC(
            vc_scheme=vc_scheme, message_resolution=args.resolution)
        img_shares = gray_hvc.encrypt(img_inputs, img_message=img_message)

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
