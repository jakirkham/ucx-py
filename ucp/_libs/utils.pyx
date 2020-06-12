# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import asyncio
import uuid
from libc.stdint cimport uintptr_t, int64_t, uint64_t
from cpython.memoryview cimport PyMemoryView_GET_BUFFER
from core_dep cimport *
from ..exceptions import UCXError, UCXCloseError


def get_buffer_data(buffer, bint check_writable=False):
    """
    Returns data pointer of the buffer. Raising ValueError if the buffer
    is read only and check_writable=True is set.
    """
    cdef dict iface = None
    if hasattr(buffer, "__cuda_array_interface__"):
        iface = buffer.__cuda_array_interface__
    elif hasattr(buffer, "__array_interface__"):
        iface = buffer.__array_interface__

    cdef uintptr_t data_ptr
    cdef bint data_readonly
    if iface is not None:
        data_ptr_obj, data_readonly = iface['data']
        # Workaround for numba giving None, rather than an 0.
        # https://github.com/cupy/cupy/issues/2104 for more info.
        if data_ptr_obj is None:
            data_ptr = 0
        else:
            data_ptr = <uintptr_t>data_ptr_obj
    else:
        mview = memoryview(buffer)
        data_ptr = <uintptr_t>PyMemoryView_GET_BUFFER(mview).buf
        data_readonly = <bint>mview.readonly

    if data_ptr == 0:
        raise NotImplementedError("zero-sized buffers isn't supported")

    if check_writable and data_readonly:
        raise ValueError("writing to readonly buffer!")

    return data_ptr


cpdef size_t get_buffer_nbytes(buffer, check_min_size, cuda_support):
    """
    Returns the size of the buffer in bytes. Returns ValueError
    if `check_min_size` is greater than the size of the buffer
    """

    iface = None
    if hasattr(buffer, "__cuda_array_interface__"):
        iface = buffer.__cuda_array_interface__
        if not cuda_support:
            raise ValueError(
                "UCX is not configured with CUDA support, please add "
                "`cuda_copy` and/or `cuda_ipc` to "
                "the UCX_TLS environment variable"
            )
    elif hasattr(buffer, "__array_interface__"):
        iface = buffer.__array_interface__

    cdef size_t i
    cdef int64_t s
    cdef size_t itemsize
    cdef uint64_t[::1] shape
    cdef int64_t[::1] strides
    cdef size_t nbytes
    if iface is not None:
        import numpy
        itemsize = numpy.dtype(iface['typestr']).itemsize
        # Making sure that the elements in shape is integers
        shape = numpy.array(iface['shape'], dtype=numpy.uint64)
        nbytes = itemsize
        for i in range(len(shape)):
            nbytes *= shape[i]
        # Check that data is contiguous
        if iface['strides'] is not None:
            strides = numpy.array(iface['strides'], dtype=numpy.int64)
            if len(shape) > 0:
                if len(strides) != len(shape):
                    raise ValueError(
                        "The length of shape and strides must be equal"
                    )
                s = <int64_t>itemsize
                for i from len(shape) >= i >= 0 by 1:
                    if s != strides[i]:
                        raise ValueError("Array must be contiguous")
                    s *= <int64_t>shape[i]
        if "mask" in iface:
            raise NotImplementedError("mask attribute not supported")
    else:
        mview = memoryview(buffer)
        nbytes = mview.nbytes
        if not mview.contiguous:
            raise ValueError("buffer must be contiguous")

    if check_min_size is not None and nbytes < check_min_size:
        raise ValueError("the nbytes is greater than the size of the buffer!")
    return nbytes
