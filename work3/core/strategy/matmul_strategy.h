#ifndef IMATMUL_STRATEGY_H
#define IMATMUL_STRATEGY_H

#include "../include/kinds.h"

template <AtomKind AtomT>
class MatrixView;

template <AtomKind AtomT>
class IMatmulStrategy {
public:
    virtual void multiply(MatrixView<AtomT> a, MatrixView<AtomT> b, 
                                MatrixView<AtomT> res) = 0;
        
    virtual ~IMatmulStrategy() = default;
};

#endif