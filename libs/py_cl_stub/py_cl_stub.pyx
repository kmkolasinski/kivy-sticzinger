# distutils: language = c++


cdef extern from 'cl_stub_main_test.h':
    int cl_stub_run_main "run_test"() nogil



cpdef run():
    return cl_stub_run_main()