#ifndef IMATMUL_STRATEGY_H
#define IMATMUL_STRATEGY_H

#include "../include/kinds.h"

template <AtomKind AtomA>
class MatrixView;

template <AtomKind AtomA, AtomKind AtomC>
class MatmulStrategy {
public:
    using Output = AtomC;
    virtual void multiply(MatrixView<AtomA> a, MatrixView<AtomA> b, 
                                MatrixView<AtomC> res) = 0;
        
    virtual ~MatmulStrategy() = default;
};

#endif