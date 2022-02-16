#pragma once

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Algorithms/reduce.h>

#include <torch/extension.h>

using namespace noa::TNL::Containers;
using namespace noa::TNL::Algorithms;

template< typename Dtype, typename Device >
inline Dtype map_reduce_tnl(const torch::Tensor &tensor)
{
    const int n = tensor.numel();
    const auto vector = VectorView< Dtype, Device>{tensor.data_ptr<Dtype>(), n};
    
    auto fetch = [=] __cuda_callable__ ( int i )-> Dtype {
            return vector[ 2 * i ]; };
    auto reduction = [] __cuda_callable__ ( const Dtype& a, const Dtype& b ) { return a + b; };
    return reduce< Device >( 0, n / 2, fetch, reduction, 0.0 );
}
