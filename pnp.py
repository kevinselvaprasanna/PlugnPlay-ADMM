# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function

import numpy as np
from scipy.sparse.linalg import LinearOperator

from bm3d import bm3d_rgb
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
from sporco import util
from sporco import plot
plot.config_notebook_plotting()

import imageio
import time
import cv2
from KAIR.denoise_dncnn import denoise_dncnn
# Define demosaicing forward operator and its transpose.

def A(x):
    """Map an RGB image to a single channel image with each pixel
    representing a single colour according to the colour filter array.
    """

    y = np.zeros(x.shape[0:2])
    y[1::2, 1::2] = x[1::2, 1::2, 0]
    y[0::2, 1::2] = x[0::2, 1::2, 1]
    y[1::2, 0::2] = x[1::2, 0::2, 1]
    y[0::2, 0::2] = x[0::2, 0::2, 2]
    return y


def AT(x):
    """Back project a single channel raw image to an RGB image with zeros
    at the locations of undefined samples.
    """

    y = np.zeros(x.shape + (3,))
    y[1::2, 1::2, 0] = x[1::2, 1::2]
    y[0::2, 1::2, 1] = x[0::2, 1::2]
    y[1::2, 0::2, 1] = x[1::2, 0::2]
    y[0::2, 0::2, 2] = x[0::2, 0::2]
    return y
# Define baseline demosaicing function. If package colour_demosaicing is installed, use the demosaicing algorithm of [37], othewise use simple bilinear demosaicing.

if have_demosaic:
    def demosaic(cfaimg):
        return demosaicing_CFA_Bayer_Menon2007(cfaimg, pattern='BGGR')
else:
    def demosaic(cfaimg):
        return bilinear_demosaic(cfaimg)

def f(x):
    return 0.5 * np.linalg.norm((A(x) - sn).ravel())**2
# Define proximal operator of data fidelity term for PPP problem.

def proxf(x, rho, tol=1e-3, maxit=100):
    ATA = lambda z: AT(A(z))
    ATAI = lambda z: ATA(z.reshape(rgbshp)).ravel() + rho * z.ravel()
    lop = LinearOperator((rgbsz, rgbsz), matvec=ATAI, dtype=s.dtype)
    b = AT(sn) + rho * x
    vx, cgit = _cg_wrapper(lop, b.ravel(), None, tol, maxit)
    return vx.reshape(rgbshp)
# Define proximal operator of (implicit, unknown) regularisation term for PPP problem. In this case we use BM3D [18] as the denoiser, using the code released with [35].


def gaussian_filter(x, rho=1):
    out = np.zeros(x.shape)
    out[:,:,0] = cv2.GaussianBlur(src=x[:,:,0],ksize=(5,5),sigmaX=rho,sigmaY=rho,borderType=cv2.BORDER_REPLICATE)
    out[:,:,1] = cv2.GaussianBlur(src=x[:,:,1],ksize=(5,5),sigmaX=rho,sigmaY=rho,borderType=cv2.BORDER_REPLICATE)
    out[:,:,2] = cv2.GaussianBlur(src=x[:,:,2],ksize=(5,5),sigmaX=rho,sigmaY=rho,borderType=cv2.BORDER_REPLICATE)
    return out

def bilateral_filter(x, rho=0.4):
    return cv2.bilateralFilter(src=np.float32(x), d=5, sigmaColor=20, sigmaSpace=rho)

def proxg(x, rho):
    #return bilateral_filter(x, bsigma)
    #return bilateral_filter(x, 0.4)
    return np.float32(denoise_dncnn(np.float32(x)))/255.
    
if __name__ == "__main__" :
    # Load reference image.

    # img = util.ExampleImages().image('kodim23.png', scaled=True,
    #                                  idxexp=np.s_[160:416,60:316])
    img_name = '/home/kvalanar/ece595-project/PlugnPlay-ADMM/kodim15.png'
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img/255.)
    # Construct test image constructed by colour filter array sampling and adding Gaussian white noise.

    np.random.seed(12345)
    s = A(img)
    rgbshp = s.shape + (3,)  # Shape of reconstructed RGB image
    rgbsz = s.size * 3       # Size of reconstructed RGB image
    nsigma = 0            # Noise standard deviation
    sn = s + nsigma * np.random.randn(*s.shape)
    
    bsigma = 0.6  # Denoiser parameter

    # Define data fidelity term for PPP problem.

    # Construct a baseline solution and initaliser for the PPP solution by BM3D denoising of a simple bilinear demosaicing solution. The 3 * nsigma denoising parameter for BM3D is chosen empirically for best performance.
    
    #imgb = demosaic(sn)
    imgb = proxf(img, rho=0)
    # Set algorithm options for PPP solver, including use of bilinear demosaiced solution as an initial solution.
    
    opt = PPP.Options({'Verbose': True, 'RelStopTol': 1e-3,
                       'MaxMainIter': 12, 'rho': 0.007, 'Y0': imgb})
    #Create solver object and solve, returning the the demosaiced image imgp.
    
    b = PPP(img.shape, f, proxf, proxg, opt=opt)
    imgp = b.solve()
    # Itn   FVal      r         s
    # ----------------------------------
    #    0  3.77e-01  2.39e-02  4.07e-01
    #    1  1.62e+00  1.97e-02  8.63e-02
    #    2  3.05e+00  1.48e-02  8.40e-02
    #    3  4.34e+00  1.13e-02  9.56e-02
    #    4  5.32e+00  9.15e-03  8.91e-02
    #    5  6.01e+00  8.08e-03  6.88e-02
    #    6  6.55e+00  7.23e-03  4.71e-02
    #    7  6.99e+00  6.19e-03  3.12e-02
    #    8  7.39e+00  5.03e-03  2.93e-02
    #    9  7.77e+00  4.06e-03  3.33e-02
    #   10  8.12e+00  3.62e-03  3.45e-02
    #   11  8.46e+00  3.52e-03  2.94e-02
    # ----------------------------------
    # Display solve time and demosaicing performance.
    
    print("PPP ADMM solve time:        %5.2f s" % b.timer.elapsed('solve'))
    print("Baseline demosaicing PSNR:  %5.2f dB" % metric.psnr(img, imgb))
    print("PPP demosaicing PSNR:       %5.2f dB" % metric.psnr(img, imgp))
    # PPP ADMM solve time:        42.84 s
    # Baseline demosaicing PSNR:  35.98 dB
    # PPP demosaicing PSNR:       37.10 dB
    # Display reference and demosaiced images.
    
    fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                            figsize=(21, 7))
    plot.imview(img, title='Reference', fig=fig, ax=ax[0])
    plot.imview(imgb, title='Baseline demoisac: %.2f (dB)' %
                metric.psnr(img, imgb), fig=fig, ax=ax[1])
    plot.imview(imgp, title='PPP demoisac: %.2f (dB)' %
                metric.psnr(img, imgp), fig=fig, ax=ax[2])
    fig.show()
    
    imageio.imwrite(f'img_{time.time()}.png', img)
    #imageio.imwrite(f'sn_{time.time()}.png', sn)
    #imageio.imwrite(f'imgb_{time.time()}.png', imgb)
    imageio.imwrite(f'imgp_{time.time()}.png', imgp)