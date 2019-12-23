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
 * Copyright (c) 2015 by Contributors
 * Copyright (c) 2017 Microsoft
 * Copyright (c) 2018 University of Maryland, College Park
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file multi_proposal-inl.h
 * \brief MultiProposalTarget Operator
*/

#ifndef MXNET_OPERATOR_MULTI_PROPOSAL_INL_H_
#define MXNET_OPERATOR_MULTI_PROPOSAL_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <iostream>
#include "operator_common.h"
#include "mshadow_op.h"


namespace mxnet {
namespace op {

namespace proposal {
enum MultiProposalOpInputs {kClsProb, kBBoxPred, kImInfo};
enum MultiProposalOpOutputs {kRoIs, kScores};
enum MultiProposalForwardResource {kTempResource};
}  // proposal

struct MultiProposalParam : public dmlc::Parameter<MultiProposalParam> {
  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  float threshold;
  int rpn_min_size;
  int batch_size;
  float bbox_scale;
  nnvm::Tuple<float> scales;
  nnvm::Tuple<float> ratios;
  int feature_stride;
  float roi_iou_thresh;
  DMLC_DECLARE_PARAMETER(MultiProposalParam) {
    float tmp[] = {0, 0, 0, 0, 0, 0, 0};
    DMLC_DECLARE_FIELD(rpn_pre_nms_top_n).set_default(6000)
    .describe("Number of top scoring boxes to keep after applying NMS to RPN proposals");
    DMLC_DECLARE_FIELD(rpn_post_nms_top_n).set_default(300)
    .describe("Overlap threshold used for non-maximum"
              "suppresion(suppress boxes with IoU >= this threshold");
    DMLC_DECLARE_FIELD(threshold).set_default(0.7)
    .describe("NMS value, below which to suppress.");
    DMLC_DECLARE_FIELD(batch_size).set_default(16)
    .describe("batch size");
    DMLC_DECLARE_FIELD(bbox_scale).set_default(1)
    .describe("Scale BBox ");
    DMLC_DECLARE_FIELD(roi_iou_thresh).set_default(0.3)
    .describe("Precomputed anchor overlap IOU, only monitor proposals with IOU>= this threshold in anchor spaces");
    DMLC_DECLARE_FIELD(rpn_min_size).set_default(4)
    .describe("Minimum height or width in proposal");
    tmp[0] = 2.0f; tmp[1] = 4.0f; tmp[2] = 7.0f; tmp[3] = 10.0f; tmp[4] = 13.0f; tmp[5] = 16.0f; tmp[6] = 24.0f;
    DMLC_DECLARE_FIELD(scales).set_default(nnvm::Tuple<float>(tmp, tmp + 7))
    .describe("Used to generate anchor windows by enumerating scales");
    tmp[0] = 0.5f; tmp[1] = 1.0f; tmp[2] = 2.0f;
    DMLC_DECLARE_FIELD(ratios).set_default(nnvm::Tuple<float>(tmp, tmp + 3))
    .describe("Used to generate anchor windows by enumerating ratios");
    DMLC_DECLARE_FIELD(feature_stride).set_default(16)
    .describe("The size of the receptive field each unit in the convolution layer of the rpn,"
              "for example the product of all stride's prior to this layer.");
  }
};

template<typename xpu>
Operator *CreateOp(MultiProposalParam param);

#if DMLC_USE_CXX11
class MultiProposalProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[cls_prob, bbox_pred, im_info, gt_boxes]";
    const TShape &dshape = in_shape->at(proposal::kClsProb);
    if (dshape.ndim() == 0) return false;


    out_shape->clear();
    // output
    out_shape->push_back(Shape2(dshape[0] * param_.rpn_post_nms_top_n, 5));
    out_shape->push_back(Shape1(dshape[0] * param_.rpn_post_nms_top_n));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MultiProposalProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "MultiProposal";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  int NumVisibleOutputs() const override {
    return 2;
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListArguments() const override {
    return {"cls_prob", "bbox_pred", "im_info"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "score"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  MultiProposalParam param_;
};  // class MultiProposalProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_MULTI_PROPOSAL_INL_H_
