#ifndef IVECSUM_STRATEGY_H
#define IVECSUM_STRATEGY_H

#include "../include/vector.cuh"

template <AtomKind AtomT>
class VectorView;

template <typename Strategy, AtomKind AtomT>
class VecsumStrategy {
public:
    void add(VectorView<AtomT> a, AtomT res) const {
       return static_cast<const Strategy*>(this)->addImpl(a, res);
    }
};

#endif