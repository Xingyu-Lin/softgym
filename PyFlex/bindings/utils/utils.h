#ifndef PYFLEX_UTILS_H
#define PYFLEX_UTILS_H


#include "utils.h"
#include "stdint.h"

#include <vector>

inline void int32_abgr_to_int8_rgba(uint32_t abgr, uint8_t &r, uint8_t &g, uint8_t &b, uint8_t &a) {
    r = (uint8_t) abgr & 255;
    g = (uint8_t) (abgr >> 8) & 255;
    b = (uint8_t) (abgr >> 16) & 255;
    a = (uint8_t) (abgr >> 24) & 255;

}

#endif //PYFLEX_UTILS_H
