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
 * Copyright (c) 2018 University of Maryland, College Park
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file mask_rcnn_target-inl.h
 * \brief MaskRcnnTarget Operator
 * \author Mahyar Najibi, Bharat Singh
*/

#ifndef MXNET_OPERATOR_MASK_RCNN_TARGET_INL_H_
#define MXNET_OPERATOR_MASK_RCNN_TARGET_INL_H_

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

namespace mask {
enum MaskRcnnTargetOpInputs {kRoIs, kMaskPolys, kMaskIds};
enum MaskRcnnOpOutputs {kMaskTargets, kMaskCls};
enum MaskRcnnTargetForwardResource {kTempSpace};
}  // end of mask namespace

struct MaskRcnnTargetParam : public dmlc::Parameter<MaskRcnnTargetParam> {
  int mask_size;
  int ignore_label;
  int batch_size;
  uint64_t workspace;
  int num_proposals;
  int max_polygon_len;
  int max_num_gts;
  int num_classes;
  int max_output_masks;
  DMLC_DECLARE_PARAMETER(MaskRcnnTargetParam) {
    DMLC_DECLARE_FIELD(batch_size).set_default(10)
    .describe("batch size");
    DMLC_DECLARE_FIELD(mask_size).set_default(28)
    .describe("Size of the generated mask for each ROI");
    DMLC_DECLARE_FIELD(ignore_label).set_default(-1)
    .describe("Ignore label in output mask");
    DMLC_DECLARE_FIELD(num_proposals).set_default(50)
    .describe("Number of proposals per image");
    DMLC_DECLARE_FIELD(max_polygon_len).set_default(500)
    .describe("Maximum possible length of a polygon");
    DMLC_DECLARE_FIELD(max_num_gts).set_default(100)
    .describe("Maximum possible number of gts per image");
    DMLC_DECLARE_FIELD(num_classes).set_default(80)
    .describe("Number of classes for mask generation");
    DMLC_DECLARE_FIELD(workspace).set_default(512).set_range(0, 8192)
      .describe("Maximum temperal workspace allowed for kTempResource");

  }
};

template<typename xpu>
Operator *CreateOp(MaskRcnnTargetParam param);

#if DMLC_USE_CXX11
class MaskRcnnTargetProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 3) << "Input:[rois, mask_polys, mask_ids]";
    const TShape &dshape = in_shape->at(mask::kMaskPolys);
    if (dshape.ndim() == 0) return false;

    out_shape->clear();
    // output
    out_shape->push_back(Shape3(dshape[0] * param_.num_proposals, param_.mask_size, param_.mask_size));
    out_shape->push_back(Shape3(dshape[0] * param_.num_proposals, param_.mask_size, param_.mask_size));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MaskRcnnTargetProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "MaskRcnnTarget";
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
    return {"rois", "mask_polys", "mask_ids"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"mask_targets", "mask_cls"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  MaskRcnnTargetParam param_;
};  // class MaskRcnnTargetProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_MASK_RCNN_TARGET_INL_H_

