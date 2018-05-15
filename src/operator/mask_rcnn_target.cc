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
    inline void convertPoly2Mask(const float* roi, const float* poly, const int mask_size, float* mask)
    {
     /* !
     Converts a polygon to a pre-defined mask wrt to an roi
     *****Inputs****
     roi: The RoI bounding box 
     poly: The polygon points the pre-defined format(see below)
     mask_size: The mask size
     *****Outputs****
     overlap: overlap of each box in boxes1 to each box in boxes2
     */
      float w = roi[3] - roi[1];
      float h = roi[4] - roi[2];
      w = std::max((float)1, w);
      h = std::max((float)1, h);
      int n_seg = poly[1];

      int offset = 2 + n_seg;
      RLE* rles;
      rlesInit(&rles, n_seg);
      for(int i = 0; i < n_seg; i++){
        int cur_len = poly[i+2];
        double* xys = new double[cur_len];
        for(int j = 0; j < cur_len; j++){
        	if (j % 2 == 0)
        		xys[j] = (poly[offset+j+1] - roi[2]) * mask_size / h;
        	else
        		xys[j] = (poly[offset+j-1] - roi[1]) * mask_size / w;


        }
        rleFrPoly(rles + i, xys, cur_len/2, mask_size, mask_size);
        delete [] xys;
        offset += cur_len;
      }
      // Decode RLE to mask
      byte* byte_mask = new byte[mask_size*mask_size*n_seg];
      rleDecode(rles, byte_mask, n_seg);
      // Flatten mask
      for(int j = 0; j < mask_size*mask_size; j++)
      {
        float cur_byte = 0;
        for(int i = 0; i< n_seg; i++){
          int offset = i * mask_size * mask_size + j;
          if(byte_mask[offset]==1){
            cur_byte = 1;
            break;
          }
        }
        mask[j] = cur_byte;
      }
      
      // Check to make sure we don't have memory leak
      rlesFree(&rles, n_seg);
      delete [] byte_mask;

    }
    inline void expandBinaryMasks2ClassMasks(const float* binary_mask, const int category, const int mask_size, \
     float* class_masks, float* mask_weights)
    {
      /* !
     Given binary masks and the classes, copy each into the correct category channel
     *****Inputs****
     binary_mask: binary masks computed
     category: category for the mask
     mask_size: mask_size
     *****Outputs****
     class_masks: output masks which has param_.num_classes channels
     */
      int offset = category * mask_size * mask_size;
      for(int i = 0; i < mask_size * mask_size; i++){
        class_masks[offset + i] = (binary_mask[i] == 1) ? 1 : 0;
        mask_weights[offset + i] = 1; 
      }
    }

}  // namespace utils

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
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 2);
    //usleep(20000000);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<cpu, 2> rois = in_data[mask::kRoIs].get<cpu, 2, real_t>(s);
    Tensor<cpu, 3> gt_masks = in_data[mask::kMaskPolys].get<cpu, 3, real_t>(s);
    Tensor<cpu, 2> mask_ids = in_data[mask::kMaskIds].get<cpu, 2, real_t>(s);
    float* crois = rois.dptr_;
    float* cgt_masks = gt_masks.dptr_;
    float* cmask_ids = mask_ids.dptr_; 
    Tensor<cpu, 4> mask_outs = out_data[mask::kMaskTargets].get<cpu, 4, real_t>(s);
    Tensor<cpu, 4> mask_weights = out_data[mask::kMaskWeights].get<cpu, 4, real_t>(s);

    float* cmask_outs = mask_outs.dptr_;
    float* cmask_weights = mask_weights.dptr_;
    int mask_mem_size = param_.batch_size*param_.num_proposals*param_.mask_size*param_.mask_size*param_.num_classes;
    for(int i = 0; i < mask_mem_size; i++) {
    	cmask_outs[i] = param_.ignore_label;
      cmask_weights[i] = 0;
    }
    // Allocate memory for binary mask
    float* binary_mask = new float[param_.mask_size*param_.mask_size];
    for(int i = 0; i < param_.batch_size * param_.num_proposals; i++){
        int mask_id = cmask_ids[i];
        if (mask_id == -1) {
          continue;
        }
        
        int imid = crois[5*i];
        int poly_offset = imid * param_.max_num_gts * param_.max_polygon_len + mask_id * param_.max_polygon_len; 
        // Convert the mask polygon to a binary mask
        mask_utils::convertPoly2Mask(crois + i * 5, cgt_masks + poly_offset, param_.mask_size, binary_mask);
        // In our poly encoding the first element is the category
        int category = (int) cgt_masks[poly_offset];
        // Expand the binary mask to a class specific mask
        int out_offset =  i * param_.mask_size * param_.mask_size * param_.num_classes;
        
        mask_utils::expandBinaryMasks2ClassMasks(binary_mask, category, param_.mask_size, \
         cmask_outs + out_offset, cmask_weights+out_offset);
    }
    delete [] binary_mask;
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
    CHECK_EQ(in_grad.size(), 3);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> grois = in_grad[mask::kRoIs].get<xpu, 2, real_t>(s);
    Tensor<xpu, 3> gmask_polys = in_grad[mask::kMaskPolys].get<xpu, 3, real_t>(s);
    Tensor<xpu, 2> gmask_ids = in_grad[mask::kMaskIds].get<xpu, 2, real_t>(s);

    Assign(grois, req[mask::kRoIs], 0);
    Assign(gmask_polys, req[mask::kMaskPolys], 0);
    Assign(gmask_ids, req[mask::kMaskIds], 0);
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
.add_argument("mask_polys", "NDArray-or-Symbol", "Polygons representing the GT masks")
.add_argument("mask_ids", "NDArray-or-Symbol", "Mask assignments.")

.add_arguments(MaskRcnnTargetParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
