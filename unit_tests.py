import torch
from components import *
import traceback
from models import *

if __name__=='__main__':
    img = torch.randn([6,3,128,128])
    classes = 9
    print(f'Test image size: {img.shape}')
    print(f'Number of classes/bands: {classes}')
    print('-------------------------------')
    print('TESTING AdaIN MODULE')
    result = False
    MOD = AdaIN(img.shape[1],True,classes)
    class_idx = torch.tensor(range(img.shape[0])).unsqueeze(-1)
    try:
        out = MOD(img, class_idx=class_idx)
        result = True
    except Exception as e:
        traceback.print_exc()
        print(e)
    print(f'Test with built in embeddings successful: {result}')

    result = False
    MOD = AdaIN(img.shape[1])
    gammas = torch.randn([img.shape[0],img.shape[1]])
    betas = torch.randn([img.shape[0], img.shape[1]])
    try:
        out = MOD(img, gammas, betas)
        result = True
    except Exception as e:
        traceback.print_exc()
        print(e)
    print(f'Test with external embedding input successful: {result}')
    print('-------------------------------')
    print('TESTING ResBlockUp MODULE')
    result = False
    MOD = ResBlockUp(32, img.shape[1],internal_embeddings=True,
            num_classes=classes)
    class_idx = torch.tensor(range(img.shape[0])).unsqueeze(-1)
    test_tens = torch.randn([img.shape[0],32, 16,16])
    try:
        out = MOD(test_tens, class_idx=class_idx)
        result=True
    except Exception as e:
        traceback.print_exc()
        print(e)
    print(f'Test with built in embeddings successful: {result}')

    result = False
    MOD = ResBlockUp(32,img.shape[1])
    gamma1 = torch.randn([img.shape[0],32])
    beta1 = torch.randn([img.shape[0],32])
    gamma2 = torch.randn([img.shape[0],img.shape[1]])
    beta2 = torch.randn([img.shape[0],img.shape[1]])
    print(f'gamma1 shape: {gamma1.shape}')
    try:
        out = MOD(test_tens, gamma1, gamma2, beta1, beta2)
        result=True
    except Exception as e:
        traceback.print_exc()
        print(e)
    print(f'Test with external embedding input successful: {result}')
    print('-------------------------------')
    print('TESTING ResBlockDown MODULE')
    result = False
    MOD = ResBlockDown(img.shape[1],32,internal_embeddings=True,
            num_classes=classes)
    class_idx = torch.tensor(range(img.shape[0])).unsqueeze(-1)
    try:
        out = MOD(img, class_idx=class_idx)
        result=True
    except Exception as e:
        traceback.print_exc()
        print(e)
    print(f'Test with built in embeddings successful: {result}')

    result = False
    MOD = ResBlockDown(img.shape[1], 32)
    gamma1 = torch.randn([img.shape[0],img.shape[1]])
    beta1 = torch.randn([img.shape[0],img.shape[1]])
    gamma2 = torch.randn([img.shape[0], 32])
    beta2 = torch.randn([img.shape[0], 32])
    try:
        out = MOD(img, gamma1=gamma1, gamma2=gamma2, beta1=beta1, beta2=beta2)
        result=True
    except Exception as e:
        traceback.print_exc()
        print(e)
    print(f'Test with external embedding input successful: {result}')
    print('-------------------------------')
    print('TESTING Encoder & Decoder MODULES')
    result_e = False
    result_d = False
    E = Encoder(img.shape[1], down_layers=4, const_layers=3, num_classes=classes)
    D = Decoder(256, up_layers=4, const_layers=3, num_classes=classes)
    try:
        out = E(img, class_idx=class_idx)
        result_e = True
        print('Encoder finished')
        out = D(out, class_idx=class_idx)
        result_d = True
    except Exception as e:
        traceback.print_exc()
        print(e)

    print(f'Test of encoder successful: {result_e}')
    print(f'Test of decoder successful: {result_d}')



