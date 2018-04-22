
#include "multi_proposal_target-inl.h"

namespace mxnet {
namespace op {
template<typename xpu>
class MultiProposalTargetGPUOp : public Operator{
 public:
  explicit MultiProposalTargetGPUOp() {
    
  }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    }

    virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    }
};

template<>
Operator* CreateOp<gpu>(MultiProposalTargetParam param) {
  return new MultiProposalTargetGPUOp<gpu>();
}

}  // namespace op
}  // namespace mxnet

