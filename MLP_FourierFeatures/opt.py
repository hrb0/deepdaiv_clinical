import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # 이미지 경로 기본값 유지
    parser.add_argument('--image_path', type=str, default='C:/Users/yumi/projects/deepdaiv_clinical/MLP_FourierFeatures/images/fox.jpg',
                        help='path to the image to reconstruct')

    # 이미지 해상도 기본값 유지
    parser.add_argument('--img_wh', nargs="+", type=int, adefault=[800, 800],
                        help='resolution (img_w, img_h) of the image')

    # 위치 인코딩 사용 여부 기본값 변경 (선택사항)
    parser.add_argument('--use_pe', default=True, action='store_true',
                        help='use positional encoding or not')

    # 아키텍처 기본값 변경
    parser.add_argument('--arch', type=str, default='relu',
                        choices=['relu', 'ff', 'siren', 'gabor', 'bacon',
                                 'gaussian', 'quadratic', 'multi-quadratic',
                                 'laplacian', 'super-gaussian', 'expsin'],
                        help='network structure')

    # 기타 인자들은 변경 없이 유지
    parser.add_argument('--act_trainable', default=False, action='store_true',
                        help='whether to train activation hyperparameter')
    parser.add_argument('--sc', type=float, default=10.,
                        help='fourier feature scale factor (std of the gaussian)')
    parser.add_argument('--omega_0', type=float, default=30.,
                        help='omega in siren')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()