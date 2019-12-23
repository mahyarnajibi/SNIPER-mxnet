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
 * \file multi_proposal.cc
 * \brief Proposal target layer
*/

#include "./multi_proposal-inl.h"
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
namespace mxnet {
namespace op {
namespace utils {

inline void BBoxTransformInv(float* boxes,
                             float* deltas,
                             float* im_info,
                             int num_images,
                             int anchors,
                             int heights,
                             int widths) {
  int num_anchors = anchors * heights * widths;
  //usleep(20000000);
  #pragma omp parallel for num_threads(8)
  for (int t = 0; t < num_images * num_anchors; ++t) {
    int b = t / num_anchors;
    int index = t % num_anchors;
    int a = index / (heights*widths);
    int mat = index % (heights*widths);
    int w = mat % widths; //width index
    int h = mat / widths; //height index
    float width = boxes[5*t + 2] - boxes[5*t] + 1.0;
    float height = boxes[5*t + 3] - boxes[5*t + 1] + 1.0;
    float ctr_x = boxes[5*t + 0] + 0.5 * (width - 1.0);
    float ctr_y = boxes[5*t + 1] + 0.5 * (height - 1.0);
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

    pred_x1 = std::max(std::min(pred_x1, im_info[3*b+1] - 1.0f), 0.0f);
    pred_y1 = std::max(std::min(pred_y1, im_info[3*b] - 1.0f), 0.0f);
    pred_x2 = std::max(std::min(pred_x2, im_info[3*b+1] - 1.0f), 0.0f);
    pred_y2 = std::max(std::min(pred_y2, im_info[3*b] - 1.0f), 0.0f);

    boxes[5*t] = pred_x1;
    boxes[5*t + 1] = pred_y1;
    boxes[5*t + 2] = pred_x2;
    boxes[5*t + 3] = pred_y2;
  }
}

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

// greedily keep the max detections (already sorted)
inline void NonMaximumSuppression(float* dets,
                                  int post_nms_top_n,
                                  int num_images,
                                  int num_anchors,
                                  int width,
                                  int height,
                                  std::vector< std::vector<int> > & final_keep_images,
                                  std::vector< std::vector<int> > & suppression_list,
                                  std::vector<float> anchors,
                                  float feature_stride,
                                  float thresh) {

  int total_anchors = num_images*num_anchors*width*height;
  int chip_anchors = num_anchors*width*height;

  float *area = new float[total_anchors];

  #pragma omp parallel for num_threads(8)
  for (int i = 0; i < total_anchors; ++i) {
    area[i] = (dets[5*i + 2] - dets[5*i + 0] + 1) * (dets[5*i + 3] - dets[5*i + 1] + 1);
  }

  int max_nms = std::min(12000, chip_anchors);

  #pragma omp parallel for num_threads(8)
  for (int i = 0; i < num_images; i++) {
    std::vector <float> sortids(chip_anchors);
    for (int j = 0; j < chip_anchors; j++) {
      sortids[j] = j;
    }
    int chip_index = i*chip_anchors;

    std::sort(sortids.begin(), sortids.end(),
        [&dets,chip_index](int i1, int i2) {
          return dets[5*(chip_index + i1) + 4] > dets[5*(chip_index + i2) + 4];
        });

    float *dbuf = new float[6*max_nms];

    //reorder for spatial locality in CPU, yo!
    for (int j = 0; j < max_nms; j++) {
      int index = i*chip_anchors + sortids[j];
      dbuf[6*j] = dets[5*index];
      dbuf[6*j+1] = dets[5*index+1];
      dbuf[6*j+2] = dets[5*index+2];
      dbuf[6*j+3] = dets[5*index+3];
      dbuf[6*j+4] = dets[5*index+4];
      dbuf[6*j+5] = area[index];
    }

    int vct = 0;

    int j = 0;
    for (; j < max_nms && vct < post_nms_top_n; j++) {
      int index = i*chip_anchors + sortids[j];
      float ix1 = dbuf[6*j];
      float iy1 = dbuf[6*j+1];
      float ix2 = dbuf[6*j+2];
      float iy2 = dbuf[6*j+3];
      float iarea = dbuf[6*j+5];

      if (dbuf[6*j+4] == -1) {
        continue;
      }

      final_keep_images[i].push_back(index);
      vct = vct + 1;
      for (int pind = j + 1; pind < max_nms; pind++) {
        if (dbuf[6*pind + 4] == -1) {
          continue;
        }
        float xx1 = std::max(ix1, dbuf[6*pind]);
        float yy1 = std::max(iy1, dbuf[6*pind + 1]);
        float xx2 = std::min(ix2, dbuf[6*pind + 2]);
        float yy2 = std::min(iy2, dbuf[6*pind + 3]);
        float w = std::max(0.0f, xx2 - xx1 + 1.0f);
        float h = std::max(0.0f, yy2 - yy1 + 1.0f);
        float inter = w * h;
        float ovr = inter / (iarea + dbuf[6*pind+5] - inter);
        if (ovr > 0.7) {
          dbuf[6*pind + 4] = -1;
        }
      }
    }
    delete [] dbuf;
  }
  delete [] area;
}

// For each index, check if its in the top max_nms, if it is calculate the iou, and set score to -1, if iou > 0.7.
  inline void compute_iou(int ind, float* dets, float* area, int max_nms, float ix1, float iy1, float ix2, float iy2, float iarea){
        if( dets[ind*5 + 4] != -1){
            float xx1 = std::max(ix1, dets[5*ind]);
            float yy1 = std::max(iy1, dets[5*ind + 1]);
            float xx2 = std::min(ix2, dets[5*ind + 2]);
            float yy2 = std::min(iy2, dets[5*ind + 3]);
            float w = std::max(0.0f, xx2 - xx1 + 1.0f);
            float h = std::max(0.0f, yy2 - yy1 + 1.0f);
            float inter = w * h;
            float ovr = inter / (iarea + area[ind] - inter);
            if (ovr > 0.7) {
              dets[5*ind + 4] = -1;
            }
          }

  }


inline void FastNonMaxSuppression(float* dets,
                                  std::vector<std::vector<int>> & anchor_iou_map,
                                  int post_nms_top_n,
                                  int num_images,
                                  int num_anchors,
                                  int width,
                                  int height,
                                  std::vector<std::vector<int>> & final_keep_images,
                                  std::vector<std::vector<int>> & suppression_list){

  int total_anchors = num_images*num_anchors*width*height;
  int chip_anchors = num_anchors*width*height;

  float* area = new float[total_anchors];

  // Pre compute area.
  #pragma omp parallel for num_threads(8)
  for (int i = 0; i < total_anchors; ++i) {
    area[i] = (dets[5*i + 2] - dets[5*i + 0] + 1) * (dets[5*i + 3] - dets[5*i + 1] + 1);
  }

  int max_nms = std::min(12000, chip_anchors);

  #pragma omp parallel for num_threads(8)
  for (int i = 0; i < num_images; i++) {
    std::vector <float> sortids(chip_anchors);
    for (int j = 0; j < chip_anchors; j++) {
      sortids[j] = j;
    }
    int chip_index = i*chip_anchors;
    auto start_sort = std::chrono::high_resolution_clock::now();
    std::nth_element(sortids.begin(), sortids.begin()+max_nms, sortids.end(),
        [&dets,chip_index](int i1, int i2) {
          return dets[5*(chip_index + i1) + 4] > dets[5*(chip_index + i2) + 4];
        });
    std::sort(sortids.begin(), sortids.begin()+max_nms,
        [&dets,chip_index](int i1, int i2) {
          return dets[5*(chip_index + i1) + 4] > dets[5*(chip_index + i2) + 4];
        });
 

    auto stop_sort = std::chrono::high_resolution_clock::now();
    auto duration_sort = std::chrono::duration_cast<std::chrono::microseconds>(stop_sort - start_sort);

    float *dbuf = new float[6*max_nms];

    //reorder for spatial locality in CPU, yo!
    // reorder indexes, for top max_nms scoring, proposals.
    for (int j = 0; j < max_nms; j++) {
      int index = i*chip_anchors + sortids[j];
      dbuf[6*j] = dets[5*index];
      dbuf[6*j+1] = dets[5*index+1];
      dbuf[6*j+2] = dets[5*index+2];
      dbuf[6*j+3] = dets[5*index+3];
      dbuf[6*j+4] = dets[5*index+4];
      dbuf[6*j+5] = area[index];
    }

    int vct = 0;
    int im_area = width * height;
    int idx_1;
    int w_1, h_1, c_1, w_2, h_2, c_2;
    int rel_w, rel_h;
//    int iarea, pind;
    std::vector<int> compute_idx;
    int chip_range = i*chip_anchors;
    // retrieve a proposal from the top ranked scoring proposals.
    int j = 0;
    auto start_loop = std::chrono::high_resolution_clock::now();
    for (; j < max_nms && vct < post_nms_top_n; j++) {
      int index = chip_range + sortids[j];
      float ix1 = dbuf[6*j];
      float iy1 = dbuf[6*j+1];
      float ix2 = dbuf[6*j+2];
      float iy2 = dbuf[6*j+3];
      float iarea = dbuf[6*j+5];

      idx_1 = sortids[j];
      int mat = idx_1 % (im_area);
      w_1 = mat % width;
      h_1 = mat / width;
      c_1 = idx_1 / im_area;

      if (dbuf[6*j+4] == -1 || dets[5*index + 4] == -1) {
        continue;
      }

      idx_1 = sortids[j];

      final_keep_images[i].push_back(index);
      vct = vct + 1;

      // retrieve the precomputed anchor overlap vector for the given proposal
      // each entry in the anchor_iou_map consists of a tuple containing relative width, relative height
      // and channel of the anchor, with which the overlap of current proposal's anchor was higher than certain
      // threshold.

      compute_idx = anchor_iou_map[c_1];
      int pind;
      for(int i = 0; i < compute_idx.size()/3; i++){
      // retrieve each entry of relative width, height and anchor channel.
        rel_w = compute_idx[i*3+0];
        rel_h = compute_idx[i*3+1];
        c_2   = compute_idx[i*3+2];
        w_2   = w_1 + rel_w;
        h_2   = h_1 + rel_h;
        pind  = chip_range + c_2*(height*width) + h_2 * width + w_2;

        if(w_2 < width && w_2 >=0 && h_2 < height && h_2 >=0){
            compute_iou(pind, dets, area, max_nms, ix1, iy1, ix2, iy2, iarea);
        }

      }
    }

    auto stop_loop = std::chrono::high_resolution_clock::now();
    auto duration_loop = std::chrono::duration_cast<std::chrono::microseconds>(stop_loop - start_loop);
    delete [] dbuf;
  }
  delete [] area;
  }

  inline void insert_anchor_iou(std::vector<std::vector<int>> &anchor_iou_dp, int anchor_type_1, int anchor_type_2, int dx, int dy){
    // anchor 1 intersection with anchor 2
    anchor_iou_dp[anchor_type_1].push_back(dx);
    anchor_iou_dp[anchor_type_1].push_back(dy);
    anchor_iou_dp[anchor_type_1].push_back(anchor_type_2);
  }

}  // namespace utils


template<typename xpu>
class MultiProposalGPUOp : public Operator{
 public:
 	float *scores;
 	float *bbox_deltas;
 	float *proposals;
 	float *im_info;
 	float *rois;
  float *out_scores;
     float* anchor_area;
      std::vector<std::vector<int>> anchor_iou_dp;
    bool is_anchor_overlap_precomputed;



  explicit MultiProposalGPUOp(MultiProposalParam param) {
    this->param_ = param;
    int batch_size = param.batch_size;    
    this->scores = new float[batch_size*21*2*200*200];
    this->bbox_deltas = new float[batch_size*21*4*200*200];
    this->proposals = new float[batch_size*21*5*200*200];
    this->im_info = new float[batch_size*3];
    this->rois = new float[param.rpn_post_nms_top_n * batch_size * 5];
    this->out_scores = new float[param.rpn_post_nms_top_n*batch_size];
    this->anchor_area = new float[21];
    this->is_anchor_overlap_precomputed = false;

  }

  ~MultiProposalGPUOp() {
    delete [] this->scores;
    delete [] this->bbox_deltas;
    delete [] this->proposals;
    delete [] this->im_info;
    delete [] this->rois;
    delete [] this->out_scores;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    bool useFastNMS = true;
    
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 2);
    
    using namespace mshadow;
    using namespace mshadow::expr;
    //clock_t t;
  	//t = clock();
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4> tscores = in_data[proposal::kClsProb].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> tbbox_deltas = in_data[proposal::kBBoxPred].get<gpu, 4, real_t>(s);
    Tensor<gpu, 2> tim_info = in_data[proposal::kImInfo].get<gpu, 2, real_t>(s);

    int num_images = tbbox_deltas.size(0);
    int num_anchors = tbbox_deltas.size(1) / 4;
    int height = tbbox_deltas.size(2);
    int width = tbbox_deltas.size(3);
    int count_anchors = num_anchors*height*width;
    int total_anchors = count_anchors * num_images;

    cudaMemcpy(scores, tscores.dptr_, total_anchors*2*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bbox_deltas, tbbox_deltas.dptr_, total_anchors*4*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(im_info, tim_info.dptr_, 3 * sizeof(float) * num_images, cudaMemcpyDeviceToHost);
    

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

    float a1_x1, a1_x2, a1_y1, a1_y2, a2_x1, a2_x2, a2_y1, a2_y2, a1_area, a2_area;
//    #pragma omp parallel for num_threads(8)
    for(int i = 0; i < num_anchors; i++){
        a1_x1 = anchors[4*i + 0];
        a1_y1 = anchors[4*i + 1];
        a1_x2 = anchors[4*i + 2];
        a1_y2 = anchors[4*i + 3];
        a1_area = (a1_x2 - a1_x1 + 1)*(a1_y2 - a1_y1 + 1);
        anchor_area[i] = a1_area;
    }

    //std::cout << "quack 3" << std::endl;
    #pragma omp parallel for num_threads(8)
    for (int t = 0; t < total_anchors; ++t) {
      int b = t / count_anchors;
      int index = t % count_anchors;
      int i = index / (height*width);
      int mat = t % (height*width);
      int k = mat % width; //width index
      int j = mat / width; //height index
      proposals[5*t] = anchors[4*i] + k * param_.feature_stride;
      proposals[5*t + 1] = anchors[4*i+1] + j * param_.feature_stride;
      proposals[5*t + 2] = anchors[4*i+2] + k * param_.feature_stride;
      proposals[5*t + 3] = anchors[4*i+3] + j * param_.feature_stride;
          // suppress anchors of type t1 for stride x%,
      if(((i+4)%7==0 || (i+2)%7==0)){
        proposals[5*t + 4] = -1;
        continue;
      }
      proposals[5*t + 4] = scores[b*count_anchors*2 + ((num_anchors + i)*height + j)*width + k];
    }

    std::vector<std::vector<int>> suppression_list(num_anchors);

    for(int i = 0; i < num_anchors; i++){
        suppression_list[i] = std::vector<int>(0);
    }
    // Precompute the anchor overlap
    if (!is_anchor_overlap_precomputed){
        for(int i = 0; i < num_anchors; i++){
            anchor_iou_dp.push_back(std::vector<int>());
        }
        
        float thresh = param_.roi_iou_thresh;
        int a1_c, a2_c;
        int feat_size = width*height;

        for(int i = 0; i < num_anchors; i++){
            a1_c = i;
            a1_x1 = anchors[4*i + 0];
            a1_y1 = anchors[4*i + 1];
            a1_x2 = anchors[4*i + 2];
            a1_y2 = anchors[4*i + 3];
            a1_area = anchor_area[i];
            for(int t = 0; t < count_anchors; t++){

              a2_c = t / (feat_size);

              int mat = t % (feat_size);
              int k = mat % width; //width index
              int j = mat / width; //height index

              a2_area = anchor_area[a2_c];

              a2_x1 = proposals[5*t+0];
              a2_y1 = proposals[5*t+1];
              a2_x2 = proposals[5*t+2];
              a2_y2 = proposals[5*t+3];
               //Compute iou between anchor 1 & anchor 2
              float xx1 = std::max(a1_x1, a2_x1);
              float yy1 = std::max(a1_y1, a2_y1);
              float xx2 = std::min(a1_x2, a2_x2);
              float yy2 = std::min(a1_y2, a2_y2);
              float w = std::max(0.0f, xx2 - xx1 + 1.0f);
              float h = std::max(0.0f, yy2 - yy1 + 1.0f);
              float inter = w * h;
              float ovr = inter / (a1_area + a2_area - inter);

              if(j==0 && k==0 && a2_c==a1_c){
                    continue;
              }


              if (ovr >= thresh){
              // +dx +dy
                utils::insert_anchor_iou(anchor_iou_dp, a1_c, a2_c, j, k);
                utils::insert_anchor_iou(anchor_iou_dp, a1_c, a2_c, -j, k);
                utils::insert_anchor_iou(anchor_iou_dp, a1_c, a2_c, -j, -k);
                utils::insert_anchor_iou(anchor_iou_dp, a1_c, a2_c, j, -k);
              }

            }
        }
        is_anchor_overlap_precomputed = true;
    }

    utils::BBoxTransformInv(proposals, bbox_deltas, im_info, num_images, num_anchors, height, width);

    utils::FilterBox(proposals, total_anchors, 3);

    std::vector <std::vector<int> > keep_images(num_images);
    for (int i = 0; i < num_images; i++) {
      keep_images[i] = std::vector<int>(0);
    }
    int rpn_post_nms_top_n = param_.rpn_post_nms_top_n;

    auto start = std::chrono::high_resolution_clock::now();


    if (useFastNMS) {
      utils::FastNonMaxSuppression(proposals, anchor_iou_dp, rpn_post_nms_top_n, num_images, num_anchors, width, height,
                                    keep_images, suppression_list);
    }
    else {
      utils::NonMaximumSuppression(proposals, rpn_post_nms_top_n, num_images, num_anchors, width, height, keep_images,
                                    suppression_list, anchors, param_.feature_stride, param_.roi_iou_thresh);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < num_images; i++) {
      int numpropsi = keep_images[i].size();
      for (int j = 0; j < numpropsi; j++) {
        int base = (i*rpn_post_nms_top_n + j);
        rois[5*base] = i;
        rois[5*base+1] = proposals[5*keep_images[i][j] + 0];
        rois[5*base+2] = proposals[5*keep_images[i][j] + 1];
        rois[5*base+3] = proposals[5*keep_images[i][j] + 2];
        rois[5*base+4] = proposals[5*keep_images[i][j] + 3];
        out_scores[base] = proposals[5*keep_images[i][j] + 4];
      }

      for (int j = numpropsi; j < rpn_post_nms_top_n; j++) {
        int base = (i*rpn_post_nms_top_n + j);
        rois[5*base+0] = i;
        rois[5*base+1] = rand() % 100;
        rois[5*base+2] = rand() % 100;
        rois[5*base+3] = 200 + rand() % 200;
        rois[5*base+4] = 200 + rand() % 200;
        out_scores[base] = 0.0;
      }
    }

    Stream<gpu> *so = ctx.get_stream<gpu>();
    Tensor<gpu,1> oscores = out_data[proposal::kScores].get<gpu, 1, real_t>(so);    
    Tensor<gpu, 2> orois = out_data[proposal::kRoIs].get<gpu, 2, real_t>(so);
    cudaMemcpy(orois.dptr_, rois, 5*sizeof(float) * num_images * rpn_post_nms_top_n, cudaMemcpyHostToDevice);
    cudaMemcpy(oscores.dptr_, out_scores, sizeof(float) * num_images * rpn_post_nms_top_n, cudaMemcpyHostToDevice);
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
    CHECK_EQ(in_grad.size(), 4);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gscores = in_grad[proposal::kClsProb].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gbbox = in_grad[proposal::kBBoxPred].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> ginfo = in_grad[proposal::kImInfo].get<xpu, 2, real_t>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[proposal::kClsProb], 0);
    Assign(gbbox, req[proposal::kBBoxPred], 0);
    Assign(ginfo, req[proposal::kImInfo], 0);
  }

 private:
  MultiProposalParam param_;
};  // class MultiProposalOp

template<>
Operator *CreateOp<gpu>(MultiProposalParam param) {
  return new MultiProposalGPUOp<gpu>(param);
}


}  // namespace op
}  // namespace mxnet
