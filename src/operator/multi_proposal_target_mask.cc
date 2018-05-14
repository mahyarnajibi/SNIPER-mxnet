/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 University of Maryland, College Park
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file multi_proposal_target.cc
 * \brief Proposal target layer
 * \author Bharat Singh
*/

#include "./multi_proposal_target_mask-inl.h"
#include <set>
#include <math.h>
#include <unistd.h>
#include <time.h>
//============================
// Bounding Box Transform Utils
//============================
namespace mxnet {
namespace op {
namespace utils {

}  // namespace utils


template<typename xpu>
class MultiProposalTargetMaskOp : public Operator{
 public:
  explicit MultiProposalTargetMaskOp(MultiProposalTargetMaskParam param) {
    this->param_ = param;
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

 private:
  MultiProposalTargetMaskParam param_;
};  // class MultiProposalOp

template<>
Operator *CreateOp<cpu>(MultiProposalTargetMaskParam param) {
  return new MultiProposalTargetMaskOp<cpu>(param);
}

Operator* MultiProposalTargetMaskProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(MultiProposalTargetMaskParam);

MXNET_REGISTER_OP_PROPERTY(MultiProposalTargetMask, MultiProposalTargetMaskProp)
.describe("Generate region proposals via RPN")
.add_argument("cls_prob", "NDArray-or-Symbol", "Score of how likely proposal is object.")
.add_argument("bbox_pred", "NDArray-or-Symbol", "BBox Predicted deltas from anchors for proposals")
.add_argument("im_info", "NDArray-or-Symbol", "Image size and scale.")
.add_argument("gt_boxes", "NDArray-or-Symbol", "Image size and scale.")
.add_argument("valid_ranges", "NDArray-or-Symbol", "Image size and scale.")
.add_arguments(MultiProposalTargetMaskParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
