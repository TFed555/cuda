#ifndef IVECSUM_STRATEGY_H
#define IVECSUM_STRATEGY_H

#include "../include/kinds.h"

template <AtomKind AtomT>
class VectorView;

template <typename T, AtomKind AtomT>
class VecsumStrategy {
public:
    void add(VectorView<AtomT> a, AtomT res) const {
       return static_cast<const T*>(this)->addImpl();
    }
};

#endif