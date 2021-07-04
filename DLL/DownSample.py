from PIL import Image
import numpy as np

'''========================= 给图像加下采样后加高斯模糊 ========================
FUNCTION:   DownSampling_n_GaussianBlur
FEATURE:    给图像加下采样后加高斯模糊
INPUTS:img,scale
       img-------------PIL图像对象(Format:Image.open('lena.bmp'))
       size_image------图像大小(Format:(512,512))
       scale-----------下采样因子(Format:2)
       gauss_radius----高斯模糊核方差(Format:1.5)
OUTPUT:img-------------下采样加模糊之后的图像对象
============================================================================='''
def DownScale_n_GaussianBlur(img,size_image,scale = 2,gauss_radius = 1.5):
    from PIL.ImageFilter import GaussianBlur
    size_im = np.array(size_image)
    img = img.resize((size_im//scale))
    img = img.resize((size_im),Image.BICUBIC)
    img = img.filter(GaussianBlur(radius = gauss_radius))
    return img

'''============================= 给图像加高斯模糊 ==============================
FUNCTION:   GaussianBlur
FEATURE:    给图像加高斯模糊
INPUTS:img,size_image,gauss_radius
       img-------------PIL图像对象(Format:Image.open('lena.bmp'))
       size_image------图像大小(Format:(512,512))
       scale-----------下采样因子(Format:2)
       gauss_radius----高斯模糊核方差(Format:1.5)
OUTPUT:img-------------下采样加模糊之后的图像对象
============================================================================='''
def GaussianBlur(img,size_image,gauss_radius = 1.5):
    return DownScale_n_GaussianBlur(img,size_image,scale = 1,gauss_radius = 1.5)

'''================================ 给图像下采样 ===============================
FUNCTION:   DownSampling
FEATURE:    给图像下采样
INPUTS:img,size_image,scale
       img---------PIL图像对象(Format:Image.open('lena.bmp'))
       size_image--图像大小(Format:(512,512))
       scale-------下采样因子(Format:2)
OUTPUT:img---------下采样加模糊之后的图像对象
============================================================================='''
def DownScale(img,size_image,scale = 2,gauss_radius = 1.5):
    size_im = np.array(size_image)
    img = img.resize((size_im//2))
    img = img.resize((size_im))
    return img

'''============================= 给图像加Gauss噪声 =============================
FUNCTION:   AddGaussNoise
FEATURE:    给图像加Gauss噪声
INPUTS:img,mu,std
       img---------PIL图像对象(Format:Image.open('lena.bmp'))
       mu----------噪声水平(Format:10)
       scale-------噪声方差(Format:1)
OUTPUT:img---------下采样加模糊之后的图像对象
============================================================================='''
def AddGaussNoise(img,mu,std = 1):
    img = np.array(img)
    noise = np.random.normal(0, std, img.shape)
    out = img + mu*noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return Image.fromarray(out)

'''============================= 给图像加下采样后加高斯模糊加Gauss噪声 =============================
FUNCTION:   AddGaussNoise
FEATURE:    给图像加Gauss噪声
INPUTS:img,mu,std
       img---------PIL图像对象(Format:Image.open('lena.bmp'))
       mu----------噪声水平(Format:10)
       scale-------噪声方差(Format:1)
OUTPUT:img---------下采样加模糊之后的图像对象
============================================================================='''
def DownScale_n_GaussianBlur_n_AddGaussNoise(img,size_image,scale = 2,gauss_radius = 1.5):
    img = DownScale_n_GaussianBlur(img,size_image,scale,gauss_radius)
    img = AddGaussNoise(img,mu = 5,std = 1)
    return img
