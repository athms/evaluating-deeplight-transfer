#!/usr/bin/python
from nilearn.masking import apply_mask, unmask
from nilearn.signal import clean


def clean_func(
    func,
    mask,
    smoothing_fwhm=3.,
    high_pass=1./128.,
    tr=0.72):
  """Clean voxel timeseries data.

  1. Smoothing with Gaussian Kernel
  2. High-pass filtering with Butterworth filter
  3. Detrending
  4. Z-standardization

  Args:
      func (niimg): func data
      mask (niimg): func brain mask]
      smoothing_fwhm (float, optional): FWHM for Gaussian kernel. Defaults to 3.
      high_pass (float, optional): High-pass threshold (in Hz). Defaults to 1./128..
      tr (float, optional): TR of func data (in seconds). Defaults to 0.72.

  Returns:
      niimg: Cleaned func data.
  """
  masked_func = apply_mask(func,
                            mask,
                            smoothing_fwhm=smoothing_fwhm,
                            ensure_finite=True)
  masked_func = clean(masked_func,
                      detrend=True,
                      standardize=True,
                      high_pass=high_pass,
                      t_r=tr,
                      ensure_finite=True)
  return unmask(masked_func, mask)