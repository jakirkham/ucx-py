# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

cpdef uintptr_t get_buffer_data(buffer, bint check_writable=False) except *
cpdef size_t get_buffer_nbytes(buffer, check_min_size, bint cuda_support) except *
