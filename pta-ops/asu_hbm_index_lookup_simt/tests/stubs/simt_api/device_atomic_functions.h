#ifndef ASU_LOOKUP_SIMT_TEST_DEVICE_ATOMIC_FUNCTIONS_H
#define ASU_LOOKUP_SIMT_TEST_DEVICE_ATOMIC_FUNCTIONS_H

template <typename T>
inline T asc_atomic_cas(T* address, T compare, T value)
{
    const T old = *address;
    if (old == compare) {
        *address = value;
    }
    return old;
}

#endif
