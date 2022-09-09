# Modified by Shangchen Zhou from: https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py
import os
import cv2
import argparse
import glob
import numpy
import torch
from PIL import Image
from matplotlib import cm

from torchvision.transforms.functional import normalize

import sys

sys.path.insert(1, './CodeFormer/')
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper


import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def set_realesrgan():
    if not torch.cuda.is_available():  # CPU
        import warnings
        warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                        'If you really want to use it, please modify the corresponding codes.',
                        category=RuntimeWarning)
        bg_upsampler = None
    else:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.realesrgan_utils import RealESRGANer
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=args.bg_tile,
            tile_pad=40,
            pre_pad=0,
            half=True)  # need to set False in CPU mode
    return bg_upsampler




def inference_gfpgan(photo=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    
    face_upsampler = None
    bg_upsampler = None

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128', '256']).to(device)
    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    # ------------------ set up FaceRestoreHelper -------------------
    face_helper = FaceRestoreHelper(
        2,
        face_size=512,
        crop_ratio=(1, 1),
        det_model = 'retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device)

    # -------------------- start to processing ---------------------
    # scan all the jpg and png images
    #for img_path in sorted(glob.glob(os.path.join(args.test_path, '*.[jp][pn]g'))):
    if True: #do just one image
        # clean all the intermediate results to process the next image
        face_helper.clean_all()
        img = numpy.array(photo)
        face_helper.read_image(img)
        # get face landmarks for each face

        num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=False, resize=640, eye_dist_threshold=5)
        print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
        face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = net(cropped_face_t, w=w, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face)

        # paste_back

        # upsample the background
        if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
        else:
                bg_img = None
        face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
        #if args.face_upsample and face_upsampler is not None: 
        if True and face_upsampler is not None: 
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=False, face_upsampler=face_upsampler)
        else:
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=False)

        im=Image.fromarray(restored_img)
        return  im, num_det_faces





def generate_faces(photo=None):
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        return inference_gfpgan(photo=init_image)