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
 * \file mask_rcnn_target.cc
 * \brief MaskRcnnTarget Operator
 * \author Mahyar Najibi, Bharat Singh
*/

#include "./mask_rcnn_target-inl.h"
#include "../coco_api/common/maskApi.h"
#include <set>
#include <math.h>
#include <unistd.h>
#include <time.h>

namespace mxnet {
namespace op {
namespace mask_utils {
    // Mask Utility Functions
    inline void getBoxesFromPolys(float* polys, float* out_bboxs){
     /* !
     Output the bounding box around each set of polygon which repesent an object
     *****Inputs****
     polys: polygons in a given image
     *****Outputs****
     out_bboxs: bounding boxes for each object based on polygons
     */
    }
    inline void getBboxOverlaps(float* boxes1, float* boxes2, float* overlaps)
    {
     /* !
     Compute the bounding box overlap between two set of boxes
     *****Inputs****
     boxes1: first set of bounding boxes
     boxes2: second set of bounding boxes
     *****Outputs****
     overlap: overlap of each box in boxes1 to each box in boxes2
     */
    }
    inline void convertPoly2Mask(float* roi, float* poly, float* mask)
    {
     /* !
     Converts a polygon to a pre-defined mask wrt to an roi
     *****Inputs****
     boxes1: first set of bounding boxes
     boxes2: second set of bounding boxes
     *****Outputs****
     overlap: overlap of each box in boxes1 to each box in boxes2
     */
    }
    inline void expandBinaryMasks2ClassMasks(float* binary_masks, float* classes, float* class_masks)
    {
      /* !
     Given binary masks and the classes, copy each into the correct category channel
     *****Inputs****
     binary_masks: binary masks computed
     classes: category for the mask
     *****Outputs****
     class_mask: output masks which has param_.num_classes channels
     */
    }


}  // namespace mask_utils


template<typename xpu>
class MaskRcnnTargetOp : public Operator{
 public:
  explicit MaskRcnnTargetOp(MaskRcnnTargetParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 5);
    CHECK_EQ(out_data.size(), 1);

    // Getting the inputs
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<cpu, 2> rois = in_data[mask::kRoIs].get<cpu, 2, real_t>(s);
    Tensor<cpu, 2> im_info = in_data[mask::kImInfo].get<cpu, 2, real_t>(s);
    Tensor<cpu, 2> gt_masks = in_data[mask::kGTMasks].get<cpu, 2, real_t>(s);
    Tensor<cpu, 3> gt_boxes = in_data[mask::kGTBoxes].get<cpu, 3, real_t>(s);
    Tensor<cpu, 2> valid_ranges = in_data[mask::kValidRanges].get<cpu, 2, real_t>(s);

    // Getting the outputs
    Tensor<cpu, 4> out_masks = out_data[mask::kMaskTargets].get<cpu, 4, real_t>(s);
    
    // The polygon format for each ground-truth object is as follows:
    // [category, num_seg, len_seg1, len_seg2,....,len_segn, seg1_x1,seg1_y1,...,seg1_xm,seg1_ym,seg2_x1,seg2_y1,...]


    // Compute the bounding box around the polygons of each object (each object may have multiple polys)
    // For each RoI find the mask bbox which have the highest overlap
    // For each RoI project back the selected mask w.r.t. the RoI and projected in the requested out size
    // Expand the binary masks to param_.num_classes channels


  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), 5);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> gscores = in_grad[mask::kRoIs].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> gbbox = in_grad[mask::kImInfo].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> ginfo = in_grad[mask::kGTMasks].get<xpu, 2, real_t>(s);
    Tensor<xpu, 3> ggt_boxes = in_grad[mask::kGTBoxes].get<xpu, 3, real_t>(s);
    Tensor<xpu, 2> gvalid_ranges = in_grad[mask::kValidRanges].get<xpu, 2, real_t>(s);

    Assign(gscores, req[mask::kRoIs], 0);
    Assign(gbbox, req[mask::kImInfo], 0);
    Assign(ginfo, req[mask::kGTMasks], 0);
    Assign(ggt_boxes, req[mask::kGTBoxes], 0);
    Assign(gvalid_ranges, req[mask::kValidRanges], 0);
  }

 private:
  MaskRcnnTargetParam param_;
};  // class MaskRcnnTargetOp

template<>
Operator *CreateOp<cpu>(MaskRcnnTargetParam param) {
  return new MaskRcnnTargetOp<cpu>(param);
}

Operator* MaskRcnnTargetProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(MaskRcnnTargetParam);

MXNET_REGISTER_OP_PROPERTY(MaskRcnnTarget, MaskRcnnTargetProp)
.describe("Generates the target segmentaion mask for each RoI")
.add_argument("rois", "NDArray-or-Symbol", "RoIs generated by RPN")
.add_argument("im_info", "NDArray-or-Symbol", "Image size and scale.")
.add_argument("gt_masks", "NDArray-or-Symbol", "Polygons representing the GT masks")
.add_argument("gt_boxes", "NDArray-or-Symbol", "The ground-truth boxes")
.add_argument("valid_ranges", "NDArray-or-Symbol", "Valid ranges for multi-scale training")
.add_arguments(MaskRcnnTargetParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
