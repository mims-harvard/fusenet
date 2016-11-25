from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy as np
from numpy.distutils.system_info import get_info

# blas_include = '/System/Library/Frameworks/Accelerate.framework/Versions/A/' \
#                 'Frameworks/vecLib.framework/Versions/A/Headers'
blas_include = get_info('blas_opt_info')['extra_compile_args'][1][2:]

cythonize('fusenet/model/cd_fast.pyx')

setup(name='Fast elastic net coordinate descent optimization algorithm',
    ext_modules=[
        Extension('fusenet/model/cd_fast', ['fusenet/model/cd_fast.c'],
                  include_dirs=[np.get_include(), blas_include])
    ]
)
