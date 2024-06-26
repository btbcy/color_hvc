# Color Extended Visual Cryptography Using Error Diffusion

Unofficial implementation of "Color extended visual cryptography using error diffusion". Secret message is encoded into multiple images (shares) by error diffusion halftoning. These shares can be combined to reveal the original message without any digital processing.

![demo](./output_image/demo.gif)

## Usage

Run the `color_hvc.py` script with the following options:

```
usage: color_hvc.py [-h] [-s SCHEME SCHEME] -ip INPUT_PATH -if INPUT_FNAMES [INPUT_FNAMES ...] [-m MESSAGE] -op
                    OUTPUT_PATH [-oef OUTPUT_FNAMES [OUTPUT_FNAMES ...]] [-r RESOLUTION RESOLUTION] [-d]
                    [-odf OUTPUT_DECRYPT_FNAME]

Color extended visual cryptography using error diffusion

optional arguments:
  -h, --help            show this help message and exit
  -s SCHEME SCHEME, --scheme SCHEME SCHEME
                        VC scheme, (2, 2) or (3, 4)
  -ip INPUT_PATH, --input_path INPUT_PATH
                        input images path
  -if INPUT_FNAMES [INPUT_FNAMES ...], --input_fnames INPUT_FNAMES [INPUT_FNAMES ...]
                        input images filenames
  -m MESSAGE, --message MESSAGE
                        message image filename
  -op OUTPUT_PATH, --output_path OUTPUT_PATH
                        output images path
  -oef OUTPUT_FNAMES [OUTPUT_FNAMES ...], --output_fnames OUTPUT_FNAMES [OUTPUT_FNAMES ...]
                        encrypted share images filenames
  -r RESOLUTION RESOLUTION, --resolution RESOLUTION RESOLUTION
                        destination message resolution to be resized
  -d, --decrypt
  -odf OUTPUT_DECRYPT_FNAME, --output_decrypt_fname OUTPUT_DECRYPT_FNAME
                        recovered message filename
```

### Example for encryption and decryption

```
python color_hvc.py -s 3 4 -ip src_image -if Lena.png Baboon.png Barbara.bmp House.bmp -m peppers.png -op output_image -r 128 128 -d -odf recover.png
```

This command encrypts the message image `peppers.png` using the images `Lena.png`, `Baboon.png`, `Barbara.bmp`, and `House.bmp` located in the `src_image` directory. The decrypted image are saved in the `output_image` directory.

### Example for encryption only

```
python color_hvc.py -s 3 4 -ip src_image -if Lena.png Baboon.png Barbara.bmp House.bmp -m peppers.png -op output_image -r 128 128
```

This command encrypts the message image `peppers.png` using the specified input images without performing decryption.

### Example for decryption only

```
python color_hvc.py -d -ip src_image -if shares_1.png shares_2.png shares_3.png -op output_image -odf recover.png
```

This command decrypts the shares `shares_1.png`, `shares_2.png`, and `shares_3.png` from the `src_image` directory to produce the recovered message in `output_image`.

## References

Kang, I., et al. (2010). "Color extended visual cryptography using error diffusion." IEEE transactions on image processing 20(1): 132-145.

## Acknowledgements

This project is based on the paper by Kang, I., et al. (2010) with modifications. Most concepts and methodologies implemented in this project are derived from their research. This is an unofficial implementation intended for educational and research purposes.
