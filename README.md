# Color Extended Visual Cryptography Using Error Diffusion

![demo](./output_image/demo.gif)
## How to run

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

will take peppers as message and lena, baboon, barbara, house as inputs under ./src_image and output to ./output_image.
Since -d flag is on, decrypted image to ./output_image/recover.png

### Example for encryption only

```
python color_hvc.py -s 3 4 -ip src_image -if Lena.png Baboon.png Barbara.bmp House.bmp -m peppers.png -op output_image -r 128 128
```

### Example for decryption only

```
python color_hvc.py -d -ip src_image -if shares_1.png shares_2.png shares_3.png -op output_image -odf recover.png
```

## References

Kang, I., et al. (2010). "Color extended visual cryptography using error diffusion." IEEE transactions on image processing 20(1): 132-145.
