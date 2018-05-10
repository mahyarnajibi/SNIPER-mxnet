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
 * \file mask_rcnn_target.cu
 * \brief MaskRcnnTarget Operator
 * \author Mahyar Najibi, Bharat Singh
*/

#include "./mask_rcnn_target-inl.h"
#include "../coco_api/common/maskApi.h"
#include <set>
#include <math.h>
#include <unistd.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include "./operator_common.h"
#include "./mshadow_op.h"
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



}  // namespace utils


template<typename xpu>
class MaskRcnnTargetGPUOp : public Operator{
 public:
 	float *mask_outs;

  explicit MaskRcnnTargetGPUOp(MaskRcnnTargetParam param) {
    this->param_ = param;
    this->mask_outs = new float[param_.max_output_masks*param_.num_proposals*param_.mask_size*param_.mask_size*param_.num_classes];
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
  	CHECK_EQ(in_data.size(), 5);
    CHECK_EQ(out_data.size(), 1);
    using namespace mshadow;
    using namespace mshadow::expr;
    
    // Get input
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 2> rois = in_data[mask::kRoIs].get<gpu, 2, real_t>(s);
    Tensor<gpu, 2> im_info = in_data[mask::kImInfo].get<gpu, 2, real_t>(s);
    Tensor<gpu, 2> gt_masks = in_data[mask::kGTMasks].get<gpu, 2, real_t>(s);
    Tensor<gpu, 3> gt_boxes = in_data[mask::kGTBoxes].get<gpu, 3, real_t>(s);
    Tensor<gpu, 2> valid_ranges = in_data[mask::kValidRanges].get<gpu, 2, real_t>(s);

    // Get output
	  Stream<gpu> *so = ctx.get_stream<gpu>();    
    Tensor<gpu, 4> out_masks = out_data[mask::kMaskTargets].get<gpu, 4, real_t>(so);
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
};  // class MaskRcnnTarget

template<>
Operator *CreateOp<gpu>(MaskRcnnTargetParam param) {
  return new MaskRcnnTargetGPUOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet