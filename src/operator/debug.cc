
#include "debug-inl.h"
#include <nnvm/op_attr_types.h>

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(DebugParam param, int dtype) {
  return new DebugOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *DebugProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(DebugParam);

MXNET_REGISTER_OP_PROPERTY(Debug, DebugProp)
.describe("Debug Layer")
.add_argument("data", "NDArray-or-Symbol", "Input data")
.add_arguments(DebugParam::__FIELDS__());

NNVM_REGISTER_OP(Debug)
.set_attr<nnvm::FSetInputVarAttrOnCompose>("FSetInputVarAttrOnCompose",
    [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
      if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
      if (index == 3) {
        var->attrs.dict["__init__"] = "[\"zero\", {}]";
      } else if (index == 4) {
        var->attrs.dict["__init__"] = "[\"one\", {}]";
      }
    });

}  // namespace op
}  // namespace mxnet
