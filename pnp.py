# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function

import numpy as np
from scipy.sparse.linalg import LinearOperator

from bm3d import bm3d_rgb, bm3d
try:
    from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
except ImportError:
    have_demosaic = False
else:
    have_demosaic = True

from sporco.linalg import _cg_wrapper
from sporco.admm.ppp import PPP
from sporco.interp import bilinear_demosaic
from sporco import metric
from sporco import plot
plot.config_notebook_plotting()

import imageio
import time
import cv2
from KAIR.denoise_dncnn import denoise_dncnn
# Define demosaicing forward operator and its transpose.

# b = False
# if b:
    
#     def A(x):
#         """Map an RGB image to a single channel image with each pixel
#         representing a single colour according to the colour filter array.
#         """
    
#         y = np.zeros(x.shape[0:2])
#         y[1::2, 1::2] = x[1::2, 1::2, 0]
#         y[0::2, 1::2] = x[0::2, 1::2, 1]
#         y[1::2, 0::2] = x[1::2, 0::2, 1]
#         y[0::2, 0::2] = x[0::2, 0::2, 2]
#         return y
    
    
#     def AT(x):
#         """Back project a single channel raw image to an RGB image with zeros
#         at the locations of undefined samples.
#         """
    
#         y = np.zeros(x.shape + (3,))
#         y[1::2, 1::2, 0] = x[1::2, 1::2]
#         y[0::2, 1::2, 1] = x[0::2, 1::2]
#         y[1::2, 0::2, 1] = x[1::2, 0::2]
#         y[0::2, 0::2, 2] = x[0::2, 0::2]
#         return y
# else:
#     def A(x):
#         """Map an RGB image to a single channel image with each pixel
#         representing a single colour according to the colour filter array.
#         """
    
#         y = np.zeros(x.shape[0:2])
#         y[2::4, 2::4] = x[2::4, 2::4, 0]
#         y[2::4, 3::4] = x[2::4, 3::4, 0]
#         y[3::4, 2::4] = x[3::4, 2::4, 0]
#         y[3::4, 3::4] = x[3::4, 3::4, 0]
#         y[0::4, 2::4] = x[0::4, 2::4, 1]
#         y[0::4, 3::4] = x[0::4, 3::4, 1]
#         y[1::4, 2::4] = x[1::4, 2::4, 1]
#         y[1::4, 3::4] = x[1::4, 3::4, 1]
#         y[2::4, 0::4] = x[2::4, 0::4, 1]
#         y[2::4, 1::4] = x[2::4, 1::4, 1]
#         y[3::4, 0::4] = x[3::4, 0::4, 1]
#         y[3::4, 1::4] = x[3::4, 1::4, 1]
#         y[0::4, 0::4] = x[0::4, 0::4, 2]
#         y[0::4, 1::4] = x[0::4, 1::4, 2]
#         y[1::4, 0::4] = x[1::4, 0::4, 2]
#         y[1::4, 1::4] = x[1::4, 1::4, 2]
#         return y
    
    
#     def AT(x):
#         """Back project a single channel raw image to an RGB image with zeros
#         at the locations of undefined samples.
#         """
    
#         y = np.zeros(x.shape + (3,))
#         y[2::4, 2::4, 0] = x[2::4, 2::4]
#         y[2::4, 3::4, 0] = x[2::4, 3::4]
#         y[3::4, 2::4, 0] = x[3::4, 2::4]
#         y[3::4, 3::4, 0] = x[3::4, 3::4]
#         y[0::4, 2::4, 1] = x[0::4, 2::4]
#         y[0::4, 3::4, 1] = x[0::4, 3::4]
#         y[1::4, 2::4, 1] = x[1::4, 2::4]
#         y[1::4, 3::4, 1] = x[1::4, 3::4]
#         y[2::4, 0::4, 1] = x[2::4, 0::4]
#         y[2::4, 1::4, 1] = x[2::4, 1::4]
#         y[3::4, 0::4, 1] = x[3::4, 0::4]
#         y[3::4, 1::4, 1] = x[3::4, 1::4]
#         y[0::4, 0::4, 2] = x[0::4, 0::4]
#         y[0::4, 1::4, 2] = x[0::4, 1::4]
#         y[1::4, 0::4, 2] = x[1::4, 0::4]
#         y[1::4, 1::4, 2] = x[1::4, 1::4]
#         return y
    

def fspecial_gauss(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

h = fspecial_gauss(9, 1)
dim = (512, 768)
eigHtH = np.abs(np.fft.fftn(h, s=dim))**2

# Define baseline demosaicing function. If package colour_demosaicing is installed, use the demosaicing algorithm of [37], othewise use simple bilinear demosaicing.

if have_demosaic:
    def demosaic(cfaimg):
        return demosaicing_CFA_Bayer_Menon2007(cfaimg, pattern='BGGR')
else:
    def demosaic(cfaimg):
        return bilinear_demosaic(cfaimg)

def f(x):
    return 0.5 * np.linalg.norm((A(x) - sn).ravel())**2

def blur(x):
    x_padded = cv2.copyMakeBorder(x, top=len(h)//2, bottom=len(h)//2, left=len(h)//2, right=len(h)//2, borderType=cv2.BORDER_WRAP)
    xf = cv2.filter2D(src=x_padded, ddepth=-1, kernel=h)
    return xf[len(h)//2:-(len(h)//2), len(h)//2:-(len(h)//2)]

def blur_f(x):
    blur_x = blur(x)
    return 0.5 * np.linalg.norm((blur_x - sn).ravel())**2
# Define proximal operator of data fidelity term for PPP problem.

bb= True
if bb:
    A = blur
    AT = blur
    
def A(x):
        return x[::2,::2,:]
    
def AT(x):
    return x.repeat(2,axis=0).repeat(2,axis=1)
        
def proxf(x, rho, tol=1e-3, maxit=100):
    ATA = lambda z: AT(A(z))
    ATAI = lambda z: ATA(z.reshape(rgbshp)).ravel() + rho * z.ravel()
    lop = LinearOperator((rgbsz, rgbsz), matvec=ATAI, dtype=s.dtype)
    b = AT(sn) + rho * x
    vx, cgit = _cg_wrapper(lop, b.ravel(), None, tol, maxit)
    return vx.reshape(rgbshp)
# Define proximal operator of (implicit, unknown) regularisation term for PPP problem. In this case we use BM3D [18] as the denoiser, using the code released with [35].

def prox_deblur(x, rho):
    Hty = blur(x)
    rhs = np.fft.fftn(Hty + rho*x, s=dim)
    return np.real(np.fft.ifftn(rhs/(eigHtH+rho),dim))

def gaussian_filter(x, rho=1):
    out = np.zeros(x.shape)
    out[:,:,0] = cv2.GaussianBlur(src=x[:,:,0],ksize=(5,5),sigmaX=rho,sigmaY=rho,borderType=cv2.BORDER_REPLICATE)
    out[:,:,1] = cv2.GaussianBlur(src=x[:,:,1],ksize=(5,5),sigmaX=rho,sigmaY=rho,borderType=cv2.BORDER_REPLICATE)
    out[:,:,2] = cv2.GaussianBlur(src=x[:,:,2],ksize=(5,5),sigmaX=rho,sigmaY=rho,borderType=cv2.BORDER_REPLICATE)
    return out

def gaussian_filter_bw(x, rho=1):
    return cv2.GaussianBlur(src=x,ksize=(5,5),sigmaX=rho,sigmaY=rho,borderType=cv2.BORDER_REPLICATE)

def bilateral_filter(x, rho=0.4):
    return cv2.bilateralFilter(src=np.float32(x), d=10, sigmaColor=10, sigmaSpace=rho)

def proxg(x, rho):
    #return bilateral_filter(x, bsigma)
    #return bilateral_filter(x, 1.3)
    #return gaussian_filter_bw(x, 1)
    return np.float32(denoise_dncnn(np.float32(x)))/255.
    #return bm3d_rgb(x, 0.3)
 
imarr = []
if __name__ == "__main__" :
    # Load reference image.
    
    psnr = []

    for i in [23, 7, 19]:
        # img = util.ExampleImages().image('kodim23.png', scaled=True,
        #                                  idxexp=np.s_[160:416,60:316])
        img_name = f'dataset/kodim_dataset/kodim{i:02}.png'
    
        
        # forward = blur
        # myf = blur_f
        # myprof = prox_deblur
        
        forward = A
        myf = f
        myprof = proxf
        
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.float32(img/255.)
        # Construct test image constructed by colour filter array sampling and adding Gaussian white noise.
    
        np.random.seed(12345)
        s = forward(img)
        rgbshp = (s.shape[0]*2, s.shape[1]*2, s.shape[2]) # Shape of reconstructed RGB image
        rgbsz = s.size*4        # Size of reconstructed RGB image
        nsigma = 0            # Noise standard deviation
        sn = s + nsigma * np.random.randn(*s.shape)
        
       
        #imgb = demosaic(sn)
        imgb = AT(sn)
        # Set algorithm options for PPP solver, including use of bilinear demosaiced solution as an initial solution.
        
        opt = PPP.Options({'Verbose': True, 'RelStopTol': 1e-3,
                           'MaxMainIter': 50, 'rho': 0.001, 'Y0': imgb})
        #Create solver object and solve, returning the the demosaiced image imgp.
        
        b = PPP(img.shape, myf, myprof, proxg, opt=opt)
        imgp = b.solve()
        
        print("PPP ADMM solve time:        %5.2f s" % b.timer.elapsed('solve'))
        print("Baseline demosaicing PSNR:  %5.2f dB" % metric.psnr(img, imgb))
        print("PPP demosaicing PSNR:       %5.2f dB" % metric.psnr(img, imgp))
        # Display reference and demosaiced images.
        
        # fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
        #                         figsize=(21, 7))
        # plot.imview(img, title='Reference', fig=fig, ax=ax[0])
        # plot.imview(imgb, title='Baseline demoisac: %.2f (dB)' %
        #             metric.psnr(img, imgb), fig=fig, ax=ax[1])
        # plot.imview(imgp, title='PPP demoisac: %.2f (dB)' %
        #             metric.psnr(img, imgp), fig=fig, ax=ax[2])
        # fig.show()
        
        #imageio.imwrite(f'img_{time.time()}.png', img)
        #imageio.imwrite(f'sn_{time.time()}.png', sn)
        #imageio.imwrite(f'imgb_{time.time()}.png', imgb)
        imgp[img>1] = 1
        img[imgp<0] = 0
        imageio.imwrite(f'imgp_{time.time()}.png', imgp)
        

        psnr.append(metric.psnr(img, imgp))
        imarr.append(imgp)
        
print(f'Mean_psnr, {np.mean(psnr)}')