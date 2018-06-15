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
#include <stdlib.h> 
//============================
// Bounding Box Transform Utils
//============================

#define NUM_THREADS_NMS 1024

namespace mxnet {
namespace op {
namespace utils {



// filter box by set confidence to zero
// * height or width < rpn_min_size
inline void FilterBox(float *dets,
                      int num_dets, float min_size) {
  #pragma omp parallel for num_threads(8)
  for (int i = 0; i < num_dets; ++i) {
    float iw = dets[5*i + 2] - dets[5*i] + 1.0f;
    float ih = dets[5*i + 3] - dets[5*i + 1] + 1.0f;
    if (iw < min_size || ih < min_size) {
      dets[5*i+0] -= min_size / 2;
      dets[5*i+1] -= min_size / 2;
      dets[5*i+2] += min_size / 2;
      dets[5*i+3] += min_size / 2;
      dets[5*i+4] = -1.0f;
    }
  }
}


inline void _MakeAnchor(float w,
                        float h,
                        float x_ctr,
                        float y_ctr,
                        std::vector<float> *out_anchors) {
  out_anchors->push_back(x_ctr - 0.5f * (w - 1.0f));
  out_anchors->push_back(y_ctr - 0.5f * (h - 1.0f));
  out_anchors->push_back(x_ctr + 0.5f * (w - 1.0f));
  out_anchors->push_back(y_ctr + 0.5f * (h - 1.0f));
}

inline void _Transform(float scale,
                       float ratio,
                       const std::vector<float>& base_anchor,
                       std::vector<float>  *out_anchors) {
  float w = base_anchor[2] - base_anchor[0] + 1.0f;
  float h = base_anchor[3] - base_anchor[1] + 1.0f;
  float x_ctr = base_anchor[0] + 0.5 * (w - 1.0f);
  float y_ctr = base_anchor[1] + 0.5 * (h - 1.0f);
  float size = w * h;
  float size_ratios = std::floor(size / ratio);
  float new_w = std::floor(std::sqrt(size_ratios) + 0.5f) * scale;
  float new_h = std::floor((new_w / scale * ratio) + 0.5f) * scale;

  _MakeAnchor(new_w, new_h, x_ctr,
             y_ctr, out_anchors);
}

// out_anchors must have shape (n, 5), where n is ratios.size() * scales.size()
inline void GenerateAnchors(const std::vector<float>& base_anchor,
                            const nnvm::Tuple<float>& ratios,
                            const nnvm::Tuple<float>& scales,
                            std::vector<float> *out_anchors) {

  for (size_t j = 0; j < ratios.ndim(); ++j) {
    for (size_t k = 0; k < scales.ndim(); ++k) {
      _Transform(scales[k], ratios[j], base_anchor, out_anchors);
    }
  }
}

// greedily keep the max detections
__global__ void NonMaximumSuppressionCu(float* idets, int post_nms_top_n, int num_images,
                                  int num_anchors, int width, int height, int max_gts, float* propsout,
                                  float* valid_ranges, float* gt_boxes, float* ids, float* dets) {
  int pre_nms_top_n = 6000;
  int i = blockIdx.x;
  int t = threadIdx.x;
  int chip_anchors = height*width*num_anchors;
  int multiplier = pre_nms_top_n;
  int num_threads = blockDim.x;
  int chip_index = i*chip_anchors;

  for (int j = t; j < pre_nms_top_n; j = j + num_threads) {
    dets[6*i*multiplier + 6*j] = idets[chip_index*6 + 6*(int)ids[chip_index + j]];
    dets[6*i*multiplier + 6*j+1] = idets[chip_index*6 + 6*(int)ids[chip_index + j]+1];
    dets[6*i*multiplier + 6*j+2] = idets[chip_index*6 + 6*(int)ids[chip_index + j]+2];
    dets[6*i*multiplier + 6*j+3] = idets[chip_index*6 + 6*(int)ids[chip_index + j]+3];
    dets[6*i*multiplier + 6*j+4] = idets[chip_index*6 + 6*(int)ids[chip_index + j]+4];
    dets[6*i*multiplier + 6*j+5] = idets[chip_index*6 + 6*(int)ids[chip_index + j]+5];
  }  
  __syncthreads();

  int vct = 0;
  __shared__ int keeps[300]; //hard coded, sorry
  chip_index = i*multiplier;


  for (int j = chip_index; j < chip_index + pre_nms_top_n && vct < post_nms_top_n; j++) {
    if (dets[6*j+4] == -1) {
      continue;
    }
    float ix1 = dets[6*j];
    float iy1 = dets[6*j+1];
    float ix2 = dets[6*j+2];
    float iy2 = dets[6*j+3];
    float iarea = dets[6*j+5];

    if (t == 0) {
      keeps[vct] = j;
    }

    vct = vct + 1;
    float xx1, xx2, yy1, yy2, w, h, inter, ovr;
    for (int pind = j + 1 + t; pind < chip_index + pre_nms_top_n; pind = pind + num_threads) {
      if (dets[6*pind + 4] == -1) {
        continue;
      } 
      xx1 = fmaxf(ix1, dets[6*pind]);
      yy1 = fmaxf(iy1, dets[6*pind + 1]);
      xx2 = fminf(ix2, dets[6*pind + 2]);
      yy2 = fminf(iy2, dets[6*pind + 3]);
      w = fmaxf(0.0f, xx2 - xx1 + 1.0f);
      h = fmaxf(0.0f, yy2 - yy1 + 1.0f);
      inter = w * h;
      ovr = inter / (iarea + dets[6*pind+5] - inter);
      if (ovr > 0.7) {
        dets[6*pind + 4] = -1;
      }
    }
    __syncthreads();
  }
  
  //set default values and assign gt boxes
  if (t < post_nms_top_n) {
    if (t < vct) {
      propsout[5*(i*post_nms_top_n + t)] = i;
      propsout[5*(i*post_nms_top_n + t) + 1] = dets[6*keeps[t]];
      propsout[5*(i*post_nms_top_n + t) + 2] = dets[6*keeps[t]+1];
      propsout[5*(i*post_nms_top_n + t) + 3] = dets[6*keeps[t]+2];
      propsout[5*(i*post_nms_top_n + t) + 4] = dets[6*keeps[t]+3];
    } else {
      propsout[5*(i*post_nms_top_n + t)] = i;
      propsout[5*(i*post_nms_top_n + t) + 1] = t % 100;
      propsout[5*(i*post_nms_top_n + t) + 2] = t % 100;
      propsout[5*(i*post_nms_top_n + t) + 3] = (t % 100) + 200;
      propsout[5*(i*post_nms_top_n + t) + 4] = (t % 100) + 200;
    }

    if (gt_boxes[5*(i*max_gts + t) + 4] != -1 && t < max_gts) {
      float x1 = gt_boxes[5*(i*max_gts + t)];
      float y1 = gt_boxes[5*(i*max_gts + t)+1];
      float x2 = gt_boxes[5*(i*max_gts + t)+2];
      float y2 = gt_boxes[5*(i*max_gts + t)+3];

      float area = (x2 - x1) * (y2 - y1);
      if (area < valid_ranges[2*i + 1]*valid_ranges[2*i + 1] && area >= valid_ranges[2*i]*valid_ranges[2*i]) {
        propsout[5*(i*post_nms_top_n + post_nms_top_n - t - 1) + 1] = x1;
        propsout[5*(i*post_nms_top_n + post_nms_top_n - t - 1) + 2] = y1;
        propsout[5*(i*post_nms_top_n + post_nms_top_n - t - 1) + 3] = x2;
        propsout[5*(i*post_nms_top_n + post_nms_top_n - t - 1) + 4] = y2;
      }
    }

  }
  __syncthreads();  
}

__global__ void getPropsCu(float* boxes,
                             float* deltas,
                             float* im_info,
                             float* anchorbuf,
                             float* scores,
                             float* valid_ranges,
                             int num_images,
                             int anchors,
                             int heights,
                             int widths,
                             int stride,
                             float*  scorebuf,
                             float* scoreids) {
  int num_anchors = anchors * heights * widths;
  int t = blockDim.x * blockIdx.x + threadIdx.x;

  if (t < num_images * num_anchors) {
    
    int b = t / num_anchors;
    int index = t % num_anchors;
    int a = index / (heights*widths);
    int mat = index % (heights*widths);
    int w = mat % widths; //width index
    int h = mat / widths; //height index
    boxes[6*t] = anchorbuf[4*a] + w * stride;
    boxes[6*t + 1] = anchorbuf[4*a+1] + h * stride;
    boxes[6*t + 2] = anchorbuf[4*a+2] + w * stride;
    boxes[6*t + 3] = anchorbuf[4*a+3] + h * stride;
    boxes[6*t + 4] = scores[b*num_anchors*2 + ((anchors + a)*heights + h)*widths + w];

    float width = boxes[6*t + 2] - boxes[6*t] + 1.0;
    float height = boxes[6*t + 3] - boxes[6*t + 1] + 1.0;
    float ctr_x = boxes[6*t + 0] + 0.5 * (width - 1.0);
    float ctr_y = boxes[6*t + 1] + 0.5 * (height - 1.0);
    float dx = deltas[b*num_anchors*4 + a*4*widths*heights + h*widths + w];
    float dy = deltas[b*num_anchors*4 + (a*4 + 1)*widths*heights + h*widths + w];
    float dw = deltas[b*num_anchors*4 + (a*4 + 2)*widths*heights + h*widths + w];
    float dh = deltas[b*num_anchors*4 + (a*4 + 3)*widths*heights + h*widths + w];
    float pred_ctr_x = dx * width + ctr_x;
    float pred_ctr_y = dy * height + ctr_y;
    float pred_w = exp(dw) * width;
    float pred_h = exp(dh) * height;
    float pred_x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
    float pred_y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
    float pred_x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
    float pred_y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

    pred_x1 = fmaxf(fminf(pred_x1, im_info[3*b+1] - 1.0f), 0.0f);
    pred_y1 = fmaxf(fminf(pred_y1, im_info[3*b] - 1.0f), 0.0f);
    pred_x2 = fmaxf(fminf(pred_x2, im_info[3*b+1] - 1.0f), 0.0f);
    pred_y2 = fmaxf(fminf(pred_y2, im_info[3*b] - 1.0f), 0.0f);
    boxes[6*t] = pred_x1;
    boxes[6*t + 1] = pred_y1;
    boxes[6*t + 2] = pred_x2;
    boxes[6*t + 3] = pred_y2;
    
    int min_size = 3;
    if ((pred_y2 - pred_y1) < min_size && (pred_x2 - pred_x1) < min_size) {
      boxes[6*t] -= min_size/2;
      boxes[6*t + 1] -= min_size/2;
      boxes[6*t + 2] += min_size/2;
      boxes[6*t + 3] += min_size/2;
      boxes[6*t + 4] = -1;
    }
    float area = (boxes[6*t + 2] - boxes[6*t]) * (boxes[6*t + 3] - boxes[6*t + 1]);
    if (area >= valid_ranges[2*b+1] * valid_ranges[2*b+1] || area < valid_ranges[2*b]*valid_ranges[2*b]) {
      boxes[6*t + 4] = -1;  
    }
    boxes[6*t + 5] = area;
    scorebuf[t] = boxes[6*t + 4];
    scoreids[t] = index;
  }
}

}  // namespace utils


template<typename xpu>
class MultiProposalTargetMaskGPUOp : public Operator{
 public:
  float *labels;
  float *bbox_targets;
  float *bbox_weights;
  float *crois;
  float *gt_boxes;
  float *out_pos_boxes;
  float *out_pos_ids;

  explicit MultiProposalTargetMaskGPUOp(MultiProposalTargetMaskParam param) {
    this->param_ = param;
    this->param_.workspace = (param_.workspace << 20) / sizeof(float);
    this->crois = new float[300*param.batch_size*5];
    this->labels = new float[300*param.batch_size];
    this->gt_boxes = new float[param.max_gts*param.batch_size*5];
    this->bbox_targets = new float[300*param.batch_size*4];
    this->bbox_weights = new float[300*param.batch_size*4];

    this->out_pos_boxes = new float[param.max_masks*param.batch_size*5];
    this->out_pos_ids = new float[param.max_masks*param.batch_size];
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    CHECK_EQ(in_data.size(), 5);
    CHECK_EQ(out_data.size(), 6);
    
    using namespace mshadow;
    using namespace mshadow::expr;
    //clock_t t;
    //t = clock();
    Stream<gpu> *s = ctx.get_stream<gpu>();

    Tensor<gpu, 4> tscores = in_data[proposal::kClsProb].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> tbbox_deltas = in_data[proposal::kBBoxPred].get<gpu, 4, real_t>(s);
    Tensor<gpu, 2> tim_info = in_data[proposal::kImInfo].get<gpu, 2, real_t>(s);
    Tensor<gpu, 3> tgt_boxes = in_data[proposal::kGTBoxes].get<gpu, 3, real_t>(s);
    Tensor<gpu, 2> tvalid_ranges = in_data[proposal::kValidRanges].get<gpu, 2, real_t>(s);
    
    int max_gts = param_.max_gts;
    Tensor<gpu, 2> rois = out_data[proposal::kRoIs].get<gpu, 2, real_t>(s);

    int rpn_post_nms_top_n = param_.rpn_post_nms_top_n;
    int num_images = tbbox_deltas.size(0);
    int num_anchors = tbbox_deltas.size(1) / 4;
    int height = tbbox_deltas.size(2);
    int width = tbbox_deltas.size(3);
    int count_anchors = num_anchors*height*width;
    int total_anchors = count_anchors * num_images;

    int bufsize = (total_anchors*8 + num_images*rpn_post_nms_top_n*5 + num_anchors*4)*sizeof(float);
    Tensor<gpu, 1> workspace = ctx.requested[proposal::kTempSpace].get_space_typed<gpu, 1, float>(Shape1(bufsize), s);

    int pre_nms_top_n = 6000;
    float* propbuf = workspace.dptr_;
    float* scorebuf = workspace.dptr_ + total_anchors*6;
    float* idbuf = workspace.dptr_ + total_anchors*7;
    float* detbuf = workspace.dptr_ + total_anchors*8;
    float* anchorbuf = workspace.dptr_ + total_anchors*8 + num_images * 6 * pre_nms_top_n;

    std::vector<float> base_anchor(4);
    //usleep(20000000);
    base_anchor[0] = 0.0;
    base_anchor[1] = 0.0;
    base_anchor[2] = param_.feature_stride - 1.0;
    base_anchor[3] = param_.feature_stride - 1.0;

    std::vector<float> anchors;
    utils::GenerateAnchors(base_anchor,
                           param_.ratios,
                           param_.scales,
                           &anchors);
    unsigned int size = num_anchors*4*sizeof(float);
    cudaMemcpy(anchorbuf, &anchors[0], size, cudaMemcpyHostToDevice);

    //call cuda kernel
    int threadsPerBlock = NUM_THREADS_NMS; 
    int numblocks = (total_anchors/threadsPerBlock) + 1;
    utils::getPropsCu<<<numblocks, threadsPerBlock>>>(propbuf, tbbox_deltas.dptr_, tim_info.dptr_, anchorbuf, tscores.dptr_,
                                                    tvalid_ranges.dptr_, num_images, num_anchors, height, width, param_.feature_stride,
                                                    scorebuf, idbuf);
    std::vector <float> tmp(total_anchors);
    std::vector<float> ids(total_anchors);
    
    cudaDeviceSynchronize();
    cudaMemcpy(&tmp[0], scorebuf, sizeof(float) * num_images * count_anchors, cudaMemcpyDeviceToHost);

    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < total_anchors; i++) {
      ids[i] = (float)(i % count_anchors);
    }
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < num_images; i++) {
      float basep = count_anchors*i;
      std::sort(ids.begin() + i*count_anchors, ids.begin() + (i+1)*count_anchors, 
          [&tmp, basep](float i1, float i2) {
            return tmp[(int)i1 + basep] > tmp[(int)i2 + basep];
          });
    }

    cudaMemcpy(idbuf, &ids[0], sizeof(float) * num_images * count_anchors, cudaMemcpyHostToDevice);
    
    utils::NonMaximumSuppressionCu<<<num_images, threadsPerBlock>>>(propbuf, rpn_post_nms_top_n, num_images, 
                                                                  num_anchors, width, height, max_gts, rois.dptr_,
                                                                  tvalid_ranges.dptr_, tgt_boxes.dptr_, idbuf, detbuf);
    cudaDeviceSynchronize();
    cudaError_t error;
    error = cudaGetLastError();
    if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
    cudaMemcpy(crois, rois.dptr_, 5*rpn_post_nms_top_n*num_images*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gt_boxes, tgt_boxes.dptr_, 5*max_gts*num_images*sizeof(float), cudaMemcpyDeviceToHost);
    
    std::vector <int> numgts_per_image(num_images);
    
    for (int i = 0; i < num_images; i++) {
      numgts_per_image[i] = 0;
      for (int j = 0; j < max_gts; j++) {
        if (gt_boxes[i*max_gts*5 + j*5 + 4] != -1) {
          numgts_per_image[i]++;
        }
      }
    }

    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < num_images; i++) {
      for (int j = 0; j < rpn_post_nms_top_n; j++) {
        int basepos = rpn_post_nms_top_n*i + j;
        labels[basepos] = 0;
        bbox_targets[4*basepos] = 1.0;
        bbox_targets[4*basepos + 1] = 1.0;
        bbox_targets[4*basepos + 2] = 1.0;
        bbox_targets[4*basepos + 3] = 1.0;

        bbox_weights[4*basepos] = 0.0;
        bbox_weights[4*basepos + 1] = 0.0;
        bbox_weights[4*basepos + 2] = 0.0;
        bbox_weights[4*basepos + 3] = 0.0;
      }
    }

    float *maxids = new float[num_images*rpn_post_nms_top_n];
    for (int i = 0; i < num_images*rpn_post_nms_top_n; i++) {
      maxids[i] = -1;
    }

    #pragma omp parallel for num_threads(8)
    for (int imid = 0; imid < num_images; imid++) {
      int tpct = 0;
      int num_gts_this_image = numgts_per_image[imid];
      //std::cout << "gtc " << num_gts_this_image << std::endl;
      int props_this_batch = rpn_post_nms_top_n;
      if (num_gts_this_image > 0) {
        float *overlaps = new float[props_this_batch * num_gts_this_image];
        float *max_overlaps = new float[props_this_batch];
        for (int i = 0; i < props_this_batch; i++) {
          max_overlaps[i] = 0;
        }
        float *max_overlap_ids = new float[props_this_batch];
        std::set <int> positive_label_ids;
        for (int i = 0; i < props_this_batch; i++) {
          max_overlap_ids[i] = 0;
        }

        for (int i = props_this_batch; i < rpn_post_nms_top_n; i++) {
          labels[imid*rpn_post_nms_top_n + i] = -1;
        }
        //get overlaps, maximum overlaps and gt labels
        for (int i = 0; i < numgts_per_image[imid]; i++) {
          float x1 = gt_boxes[imid*5*max_gts + i*5];
          float y1 = gt_boxes[imid*5*max_gts + i*5 + 1];
          float x2 = gt_boxes[imid*5*max_gts + i*5 + 2];
          float y2 = gt_boxes[imid*5*max_gts + i*5 + 3];
          int pbase;
          float a1 = (x2 - x1) * (y2 - y1);
          float xx1, yy1, xx2, yy2, w, h, inter, ovr, a2;
          for (int j = 0; j < props_this_batch; j++) {
            pbase = rpn_post_nms_top_n*imid + j;
            xx1 = std::max(x1, crois[pbase*5 + 1]);
            yy1 = std::max(y1, crois[pbase*5 + 2]);
            xx2 = std::min(x2, crois[pbase*5 + 3]);
            yy2 = std::min(y2, crois[pbase*5 + 4]);
            w = std::max(0.0f, xx2 - xx1 + 1.0f);
            h = std::max(0.0f, yy2 - yy1 + 1.0f);
            a2 = (crois[pbase*5 + 3] - crois[pbase*5 + 1]) * (crois[pbase*5 + 4] - crois[pbase*5 + 2]);
            inter = w * h;
            ovr = inter / (a1 + a2 - inter);
            overlaps[i*num_gts_this_image + j] = ovr;

            if (overlaps[i*num_gts_this_image + j] > max_overlaps[j] && overlaps[i*num_gts_this_image + j] > 0.5) {
              max_overlaps[j] = overlaps[i*num_gts_this_image + j];
              max_overlap_ids[j] = i;
              //set labels for positive proposals
              labels[imid*rpn_post_nms_top_n + j] = gt_boxes[imid*5*max_gts + i*5 + 4];
              positive_label_ids.insert(j);
              tpct = tpct + 1;
            }
          }
        }
        //p is for proposal and g is for gt, cx is x center and w,h is width and height
        int pid, gtid;
        float gx1, gx2, gy1, gy2, px1, px2, py1, py2;
        float gcx, gcy, gw, gh, pcx, pcy, pw, ph;
        //generate bbox targets for the positive labels
        for (auto it = positive_label_ids.begin(); it !=positive_label_ids.end(); it++) {
          pid = *it;
          int baseid = (imid*rpn_post_nms_top_n + pid);
          bbox_weights[baseid*4] = 1;
          bbox_weights[baseid*4+1] = 1;
          bbox_weights[baseid*4+2] = 1;
          bbox_weights[baseid*4+3] = 1;

          gtid = max_overlap_ids[pid];

          maxids[baseid] = gtid;

          gx1 = gt_boxes[imid*5*max_gts + gtid*5];
          gy1 = gt_boxes[imid*5*max_gts + gtid*5 + 1];
          gx2 = gt_boxes[imid*5*max_gts + gtid*5 + 2];
          gy2 = gt_boxes[imid*5*max_gts + gtid*5 + 3];

          gw = gx2 - gx1 + 1;
          gh = gy2 - gy1 + 1;
          gcx = gx1 + gw*0.5;
          gcy = gy1 + gh*0.5;

          px1 = crois[baseid*5 + 1];
          py1 = crois[baseid*5 + 2];
          px2 = crois[baseid*5 + 3];
          py2 = crois[baseid*5 + 4];

          pw = px2 - px1 + 1;
          ph = py2 - py1 + 1;
          pcx = px1 + (pw-1)*0.5;
          pcy = py1 + (ph-1)*0.5;

          bbox_targets[4*baseid] =  10 * (gcx - pcx) / (pw + 1e-7);
          bbox_targets[4*baseid+1] =  10 * (gcy - pcy) / (ph + 1e-7);
          bbox_targets[4*baseid+2] =  5 * log(gw/(pw + 1e-7));
          bbox_targets[4*baseid+3] =  5 * log(gh/(ph + 1e-7));
        }
        delete [] max_overlap_ids;
        delete [] overlaps;
        delete [] max_overlaps;
      }      
    }

    int mask_ct = 0;
    for (int i = 0; i < num_images*300; i++) {
      if (labels[i] > 0 and mask_ct < num_images * param_.max_masks) {
        out_pos_boxes[5*mask_ct] = crois[5*i];
        out_pos_boxes[5*mask_ct+1] = crois[5*i+1];
        out_pos_boxes[5*mask_ct+2] = crois[5*i+2];
        out_pos_boxes[5*mask_ct+3] = crois[5*i+3];
        out_pos_boxes[5*mask_ct+4] = crois[5*i+4];
        out_pos_ids[mask_ct] = maxids[i];
        mask_ct++;
      } 
    }

    for (int i = mask_ct; i < num_images*param_.max_masks; i++) {
      out_pos_boxes[5*i] = i % num_images;
      out_pos_boxes[5*i+1] = i % 200;
      out_pos_boxes[5*i+2] = i % 200;
      out_pos_boxes[5*i+3] = i % 200 + 100;
      out_pos_boxes[5*i+4] = i % 200 + 100;
      out_pos_ids[i] = -1;
    }
    
    delete [] maxids;

    Stream<gpu> *so = ctx.get_stream<gpu>();    
    Tensor<gpu, 2> olabels = out_data[proposal::kLabels].get<gpu, 2, real_t>(so);
    Tensor<gpu, 2> obbox_targets = out_data[proposal::kBboxTarget].get<gpu, 2, real_t>(so);
    Tensor<gpu, 2> obbox_weights = out_data[proposal::kBboxWeight].get<gpu, 2, real_t>(so);
    Tensor<gpu, 2> omaskrois = out_data[proposal::kMaskRoIs].get<gpu, 2, real_t>(s);
    Tensor<gpu, 2> omaskids = out_data[proposal::kMaskIds].get<gpu, 2, real_t>(s);

    cudaMemcpy(omaskrois.dptr_, out_pos_boxes, sizeof(float) * num_images * param_.max_masks * 5, cudaMemcpyHostToDevice);
    cudaMemcpy(omaskids.dptr_, out_pos_ids, sizeof(float) * num_images * param_.max_masks, cudaMemcpyHostToDevice);
    cudaMemcpy(olabels.dptr_, labels, sizeof(float) * num_images*300, cudaMemcpyHostToDevice);
    cudaMemcpy(obbox_targets.dptr_, bbox_targets, 4*sizeof(float) * num_images*300, cudaMemcpyHostToDevice);
    cudaMemcpy(obbox_weights.dptr_, bbox_weights, 4*sizeof(float) * num_images*300, cudaMemcpyHostToDevice);    
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
    Tensor<xpu, 4> gscores = in_grad[proposal::kClsProb].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gbbox = in_grad[proposal::kBBoxPred].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> ginfo = in_grad[proposal::kImInfo].get<xpu, 2, real_t>(s);
    Tensor<xpu, 3> ggt_boxes = in_grad[proposal::kGTBoxes].get<xpu, 3, real_t>(s);
    Tensor<xpu, 2> gvalid_ranges = in_grad[proposal::kValidRanges].get<xpu, 2, real_t>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[proposal::kClsProb], 0);
    Assign(gbbox, req[proposal::kBBoxPred], 0);
    Assign(ginfo, req[proposal::kImInfo], 0);
    Assign(ggt_boxes, req[proposal::kGTBoxes], 0);
    Assign(gvalid_ranges, req[proposal::kValidRanges], 0);
  }

 private:
  MultiProposalTargetMaskParam param_;
};  // class MultiProposalOp

template<>
Operator *CreateOp<gpu>(MultiProposalTargetMaskParam param) {
  return new MultiProposalTargetMaskGPUOp<gpu>(param);
}


}  // namespace op
}  // namespace mxnet
