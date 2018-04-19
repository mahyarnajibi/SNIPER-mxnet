
#include "debug-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(DebugParam param, int dtype) {
  return new DebugOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

