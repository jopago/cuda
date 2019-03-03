#ifndef _DAUBECHIES4_H
#define _DAUBECHIES4_H

/* Definition of coefficients for the Daubechies 4 wavelet */

#define sqrt3   1.73205080757
#define daub    5.65685424949

/* Device constants */

__constant__ const double h[4] = {
    (1 + sqrt3)/daub, (3 + sqrt3)/daub,
    (3 - sqrt3)/daub, (1 - sqrt3)/daub
};

__constant__ const double g[4] = {
    (1 - sqrt3)/daub, -(3 - sqrt3)/daub, (3 + sqrt3)/daub, -(1 + sqrt3)/daub
};

/* Host constants */

const double _h[4] = {
    (1 + sqrt3)/daub, (3 + sqrt3)/daub,
    (3 - sqrt3)/daub, (1 - sqrt3)/daub
};

const double _g[4] = {
    (1 - sqrt3)/daub, -(3 - sqrt3)/daub, (3 + sqrt3)/daub, -(1 + sqrt3)/daub
};

const double _ih[4] = {
    _h[2],_g[2],_h[0],_g[0]
};

const double _ig[4] = {
    _h[3],_g[3],_h[1],_g[1]
};

#endif 