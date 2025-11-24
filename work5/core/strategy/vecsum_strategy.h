#ifndef IVECSUM_STRATEGY_H
#define IVECSUM_STRATEGY_H

#include "../include/vector.cuh"

template <AtomKind AtomT>
class VectorView;

template <template<AtomKind> class Derived, AtomKind AtomT>
class VecsumStrategy {
public:
    static void add(VectorView<AtomT> a, AtomT* res) {
       Derived<AtomT>::addImpl(a, res);
    }
};

#endif