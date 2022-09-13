import argparse
import cv2
import glob
import os

import sys
#sys.path.insert(1, './Real-ESRGAN/')
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

#'Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | realesr-animevideov3'
# realesrgan | bicubic
def realesrgan_henance(input=None, outscale=4,model_name="RealESRGAN_x4plus",alpha_upsampler="realesrgan",fp32 = False, face_enhance = True):



    tile = 0            #parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    tile_pad=10         #parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    pre_pad = 0         #parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    ext='auto'          #parser.add_argument(        '--ext',        type=str,        default='auto',        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    gpu_id=0 #on none?

    # determine models according to model names
    model_name = model_name.split('.')[0]
    if model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name in ['RealESRGAN_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif model_name in ['realesr-animevideov3']:  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('realesrgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {model_name} does not exist.')

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id)

    if face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    #os.makedirs(output, exist_ok=True)



    img = input #cv2.imread(path, cv2.IMREAD_UNCHANGED)
################################  
      
      

    #if len(img.shape) == 3 and img.shape[2] == 4:
    #        img_mode = 'RGBA'
    #else:
    #        img_mode = None


    try:
            if face_enhance:
                print("Image enhancing and face fixing")
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                print("Image enhancing")
                output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            
            
    # add me        
    #restored_img = gfpgan.gfpgan_fix_faces(np.array(image, dtype=np.uint8))
    #res = Image.fromarray(restored_img)
    #if GFPGAN_strength < 1.0:
    #   res = Image.blend(image, res, GFPGAN_strength)
    #image = res
    
    return(output)






