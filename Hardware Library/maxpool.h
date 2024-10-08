/******************************************************************************
 *  Copyright (c) 2019, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
 
/***********************************************************************************
*
*   The following code is partially derived from the code provided by Xilinx, Inc.
*   in https://github.com/Xilinx/finn-hlslib.
*
*   Significant modifications and additions have been made to the original code to
*   suit the specific needs of SustainML project https://sustainml.eu.
*
*   The following code has been developed for a work package "Hardware Architectures
*   for Low Power ML" contributing to project SustainML (Application Aware, Life-Cycle
*   Oriented Model-Hardware Co-Design Framework for Sustainable, Energy Efficient ML
*   Systems). The project has received funding from European Union’s Horizon Europe
*   research and innovation programme (HORIZON-CL4-2021-HUMAN-01)
*   under grant agreement No 101070408.
*
 **********************************************************************************/

/***********************************************************************************
 *  Copyright (C) 2024, University of Kaiserslautern-Landau (RPTU), Kaiserslautern
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ************************************************************************************/

/***********************************************************************************
 *
 *  Authors: Vladimir Rybalkin <rybalkin@rptu.de>
 *           Mohamed Moursi <mmoursi@rptu.de>
 *
 *  \file maxpool.hpp
 *
 *  Library of templated HLS functions for DNN deployment.
 *  This file lists a set of convenience funtions used to implement
 *  pooling operations.
 *
 *
 ************************************************************************************/


#ifndef MAXPOOL_H
#define MAXPOOL_H
 
#include <limits>

#include "interpret.hpp"
#include "pool.hpp"

/**
 * \brief   Max Pool implementation for Binarized values 
 * 
 * This function performes the maxpool for binary inputs, and works with kernel and stride being equal 
 *
 * \tparam     ImgDim       Width and Heigth of the Input Feature Map (assumed square)
 * \tparam     PoolDim      Dimension of the Max Pool kernel (assumed square)
 * \tparam     NumChannels  Number of Input Feature Maps
 *
 * \param      in           Input stream
 * \param      out          Output stream
 *
 */
template<unsigned int ImgDim, unsigned int PoolDim, unsigned int NumChannels>
void StreamingMaxPool(stream<ap_uint<NumChannels> > & in,
        stream<ap_uint<NumChannels> > & out) {
  CASSERT_DATAFLOW(ImgDim % PoolDim == 0);
  // need buffer space for a single maxpooled row of the image
  ap_uint<NumChannels> buf[ImgDim / PoolDim];
  for(unsigned int i = 0; i < ImgDim / PoolDim; i++) {
#pragma HLS UNROLL
    buf[i] = 0;
  }

  for (unsigned int yp = 0; yp < ImgDim / PoolDim; yp++) {
    for (unsigned int ky = 0; ky < PoolDim; ky++) {
      for (unsigned int xp = 0; xp < ImgDim / PoolDim; xp++) {
#pragma HLS PIPELINE II=1
        ap_uint<NumChannels> acc = 0;
        for (unsigned int kx = 0; kx < PoolDim; kx++) {
          acc = acc | in.read();
        }
        // pool with old value in row buffer
        buf[xp] |= acc;
      }
    }
    for (unsigned int outpix = 0; outpix < ImgDim / PoolDim; outpix++) {
#pragma HLS PIPELINE II=1
      out.write(buf[outpix]);
      // get buffer ready for next use
      buf[outpix] = 0;
    }
  }
}

/**
 * \brief   Max Pool implementation for Binarized values on multiple images
 *
 * This function performes the maxpool for binary inputs, and works with kernel and stride being equal 
 * 
 * \tparam ImgDim       Width and Heigth of the Input Feature Map (assumed square)
 * \tparam PoolDim      Dimension of the Max Pool kernel (assumed square)
 * \tparam NumChannels  Number of Input Feature Maps
 *
 * \param in            Input stream
 * \param out           Output stream
 * \param numReps       Number of time the function has to be repeatedly executed (e.g. number of images)
 *
 */
template<unsigned int ImgDim, unsigned int PoolDim, unsigned int NumChannels>
void StreamingMaxPool_Batch(stream<ap_uint<NumChannels> > & in,
        stream<ap_uint<NumChannels> > & out, unsigned int numReps) {
  for (unsigned int rep = 0; rep < numReps; rep++) {
    StreamingMaxPool<ImgDim, PoolDim, NumChannels>(in, out);
  }
}


/**
 * \brief   Max Pool implementation for non Binarized values 
 *
 * This function performes the maxpool for non-binary inputs, and works with kernel and stride being equal 
 * 
 * \tparam ImgDim       Width and Heigth of the Input Feature Map (assumed square)
 * \tparam PoolDim      Dimension of the Max Pool kernel (assumed square)
 * \tparam NumChannels  Number of Input Feature Maps
 * \tparam ActType      DataType of the input activation (as used in the comparison)
 * \tparam min_value    Minimum value possible with the given ActType, used to initialize the value before the comparison
 * \tparam StreamW      Width of the input and output stream
 * 
 * \param in            Input stream
 * \param out           Output stream
 *
 */
template
<
unsigned int ImgDim,
unsigned int PoolDim,
unsigned int NumChannels,
typename ActType,
int min_value, 
int StreamW 
>
void StreamingMaxPool_Precision(stream<ap_uint<StreamW> > & in,
        stream<ap_uint<StreamW> > & out) {
  CASSERT_DATAFLOW(ImgDim % PoolDim == 0);
  // need buffer space for a single maxpooled row of the image
  ActType buf[ImgDim / PoolDim][NumChannels];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=2
  for(unsigned int i = 0; i < ImgDim / PoolDim; i++) {
    for(unsigned int ch = 0; ch<NumChannels; ch++){
#pragma HLS UNROLL
      buf[i][ch] = min_value; //std::numeric_limits<ActType>::min();
    }
  }
  ap_uint<StreamW> inputData,outputData;
  for (unsigned int yp = 0; yp < ImgDim / PoolDim; yp++) {
    for (unsigned int ky = 0; ky < PoolDim; ky++) {
      for (unsigned int xp = 0; xp < ImgDim / PoolDim; xp++) {
        // Change to comparator 
        for (unsigned int kx = 0; kx < PoolDim; kx++) {
#pragma HLS PIPELINE II=1
          inputData = in.read();
          for(unsigned int ch = 0; ch<NumChannels; ch++){
#pragma HLS UNROLL                      
            unsigned int lowBit = ch * ActType::width;
            unsigned int highBit = (ch+1) * ActType::width -1;
            ActType channeldata = inputData(highBit, lowBit);                   
            ActType oldMax = buf[xp][ch];               
            if(channeldata > oldMax){
              buf[xp][ch] = channeldata;
            }
          }
        }
      }
    }
    for (unsigned int outpix = 0; outpix < ImgDim / PoolDim; outpix++) {
      for(unsigned int ch = 0; ch < NumChannels; ch++){
#pragma HLS UNROLL
        unsigned int lowBit = ch * ActType::width;
        unsigned int highBit = (ch+1) * ActType::width -1;  
        outputData(highBit, lowBit) = buf[outpix][ch];
        // get buffer ready for next use
        buf[outpix][ch] = min_value;
      }
      out.write(outputData);
    }
  }
}

/**
 * \brief   Max Pool implementation for non binarized values on multiple images
 *
 * This function performes the maxpool for non binary inputs, and works with kernel and stride being equal 
 * 
 * \tparam ImgDim       Width and Heigth of the Input Feature Map (assumed square)
 * \tparam PoolDim      Dimension of the Max Pool kernel (assumed square)
 * \tparam NumChannels  Number of Input Feature Maps
 * \tparam ActType      DataType of the input activation (as used in the comparison)
 * \tparam min_value    Minimum value possible with the given ActType, used to initialize the value before the comparison
 * \tparam StreamW      Width of the input and output stream
 * 
 * \param in            Input stream
 * \param out           Output stream
 * \param numReps       Number of time the function has to be repeatedly executed (e.g. number of images)
 *
 */
template
<
unsigned int ImgDim,
unsigned int PoolDim,
unsigned int NumChannels,
typename ActType,
int min_value, 
// safely deducible (stream width must be int though!)
int InStreamW,
int OutStreamW  
>
void StreamingMaxPool_Precision_Batch(stream<ap_uint<InStreamW> > & in,
        stream<ap_uint<OutStreamW> > & out, unsigned int numReps) {
#pragma HLS INLINE
  unsigned const  InpPerImage = ImgDim*ImgDim*NumChannels*ActType::width/InStreamW ;
  unsigned const  OutPerImage = ImgDim*ImgDim / (PoolDim*PoolDim);
  WidthAdjustedInputStream <InStreamW, NumChannels*ActType::width, InpPerImage>  wa_in (in,  numReps);
  WidthAdjustedOutputStream<NumChannels*ActType::width,  OutStreamW, OutPerImage>  wa_out(out, numReps);
  for (unsigned int rep = 0; rep < numReps; rep++) {
    StreamingMaxPool_Precision<ImgDim, PoolDim, NumChannels, ActType, min_value>
      (static_cast<hls::stream<ap_uint<NumChannels*ActType::width>>&>(wa_in), 
      static_cast<hls::stream<ap_uint<NumChannels*ActType::width>>&>(wa_out));
  }
}


/**
 * \brief   1D Max Pool implementation for non Binarized values 
 *
 * This function performes the maxpool for non-binary inputs, and works with kernel and stride being equal 
 * 
 * \tparam ImgDim       Length of the Input Feature Map
 * \tparam PoolDim      Dimension of the Max Pool kernel
 * \tparam NumChannels  Number of Input Feature Maps
 * \tparam ActType      DataType of the input activation (as used in the comparison)
 * \tparam min_value    Minimum value possible with the given ActType, used to initialize the value before the comparison
 * \tparam StreamW      Width of the input and output stream
 * 
 * \param in            Input stream
 * \param out           Output stream
 *
 */

template<unsigned int ImgDim, unsigned int PoolDim, unsigned int NumChannels, typename ActType, int min_value, 
        int StreamW 
        >
void StreamingMaxPool_Precision_1d(stream<ap_uint<StreamW> > & in,
        stream<ap_uint<StreamW> > & out) {
  CASSERT_DATAFLOW(ImgDim % PoolDim == 0);
  // need buffer space for a single maxpooled pixel of the image
  ActType buf[NumChannels];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1
  for(unsigned int ch = 0; ch<NumChannels; ch++){
#pragma HLS UNROLL
    buf[ch] = min_value; //std::numeric_limits<ActType>::min();
  }

  ap_uint<StreamW> inputData,outputData;
  for (unsigned int xp = 0; xp < ImgDim / PoolDim; xp++) {
    // Change to comparator 
    for (unsigned int kx = 0; kx < PoolDim; kx++) {
#pragma HLS PIPELINE II=1
      inputData = in.read();
      for(unsigned int ch = 0; ch<NumChannels; ch++){
#pragma HLS UNROLL                      
        unsigned int lowBit = ch * ActType::width;
        unsigned int highBit = (ch+1) * ActType::width -1;
        ActType channeldata = inputData(highBit, lowBit);                   
        ActType oldMax = buf[ch];               
        if(channeldata > oldMax){
          buf[ch] = channeldata;
        }
      }
    }
    for(unsigned int ch = 0; ch < NumChannels; ch++){
#pragma HLS UNROLL
      unsigned int lowBit = ch * ActType::width;
      unsigned int highBit = (ch+1) * ActType::width -1;  
      outputData(highBit, lowBit) = buf[ch];
      // get buffer ready for next use
      buf[ch] = min_value;
    }
    out.write(outputData);
  }
}

/**
 * \brief   1D Max Pool implementation for non binarized values on multiple images
 *
 * This function performes the maxpool for non binary inputs, and works with kernel and stride being equal 
 * 
 * \tparam ImgDim       Length of the Input Feature Map
 * \tparam PoolDim      Dimension of the Max Pool kernel
 * \tparam NumChannels  Number of Input Feature Maps
 * \tparam ActType      DataType of the input activation (as used in the comparison)
 * \tparam min_value    Minimum value possible with the given ActType, used to initialize the value before the comparison
 * \tparam StreamW      Width of the input and output stream
 * 
 * \param in            Input stream
 * \param out           Output stream
 * \param numReps       Number of time the function has to be repeatedly executed (e.g. number of images)
 *
 */
template<unsigned int ImgDim, unsigned int PoolDim, unsigned int NumChannels, typename ActType, int min_value, 
        int InStreamW, int OutStreamW  // safely deducible (stream width must be int though!)
        >
void StreamingMaxPool_Precision_Batch_1d(stream<ap_uint<InStreamW> > & in,
        stream<ap_uint<OutStreamW> > & out, unsigned int numReps) {
#pragma HLS INLINE
  unsigned const  InpPerImage = ImgDim*NumChannels*ActType::width/InStreamW ;
  unsigned const  OutPerImage = ImgDim / PoolDim;
  WidthAdjustedInputStream <InStreamW, NumChannels*ActType::width, InpPerImage>  wa_in (in,  numReps);
  WidthAdjustedOutputStream<NumChannels*ActType::width,  OutStreamW, OutPerImage>  wa_out(out, numReps);
  for (unsigned int rep = 0; rep < numReps; rep++) {
    StreamingMaxPool_Precision_1d<ImgDim, PoolDim, NumChannels, ActType, min_value>
      (static_cast<hls::stream<ap_uint<NumChannels*ActType::width>>&>(wa_in), 
      static_cast<hls::stream<ap_uint<NumChannels*ActType::width>>&>(wa_out));
  }
}


/**
 * \brief   ReLU for fixed-point or integer; can accept a bias at input, which it removes
 *
 * \tparam ImgDim       Width and Heigth of the Input Feature Map (assumed square)
 * \tparam NumChannels  Number of Input Feature Maps
 * \tparam ActType      DataType of the input activation (as used in the comparison)
 * \tparam PECount      PE parallelism to apply ReLU
 * \tparam offset       Offset to be subtracted before applying ReLU
 * 
 * \param in            Input stream
 * \param out           Output stream
 * \param numReps       Number of time the function has to be repeatedly executed (e.g. number of images)
 *
 */
template<
        unsigned int ImgDim,            
    unsigned int NumChannels,  
        typename ActType,           
        unsigned int PECount,
    int offset = 0>
void ReLU_Batch(stream<ap_uint<PECount * ActType::width> > & in,
        stream<ap_uint<PECount * ActType::width> > & out, const unsigned int numReps) {

    ap_uint<PECount * ActType::width> thin;
    ap_uint<PECount * ActType::width> thout;
    
    //call to thresholding library function
    for(unsigned int reps=0; reps<numReps; reps++){
        for(unsigned int pixel=0; pixel<ImgDim*ImgDim; pixel++){
      for(unsigned int fold=0; fold<NumChannels/PECount; fold++){
#pragma HLS PIPELINE II=1
        thin = in.read();
        for(unsigned int pe=0; pe<PECount; pe++){
        #pragma HLS UNROLL
          // Threshold and assign to right bits of output buffers
          unsigned int lowBit = pe * ActType::width;
          unsigned int highBit = (pe+1) * ActType::width - 1;
          ActType val = thin(highBit,lowBit);
          ActType result;
          if(val < offset)
                  result = 0;
          else
                  result = val - offset;
          thout(highBit, lowBit) = result;
        }    
        out.write(thout);
      }
        }
    }
}

/**
 * \brief   Accumulate-pool - like average pooling over the whole frame, but without the division at end
 *
 * \tparam ImgDim       Width and Heigth of the Input Feature Map (assumed square)
 * \tparam NumChannels  Number of Input Feature Maps
 * \tparam ActType      DataType of the input activation (as used in the comparison)
 * \tparam PECount      PE parallelism to apply ReLU
 * \tparam AccType      Datatype of the accumulation (e.g. output)
 * 
 * \param in            Input stream
 * \param out           Output stream
 * \param numReps       Number of time the function has to be repeatedly executed (e.g. number of images)
 *
 */
template<
    unsigned int ImgDim,     
        unsigned int NumChannels,       
        typename ActType,           
        unsigned int PECount,      
        typename AccType>
void AccPool_Batch(stream<ap_uint<PECount * ActType::width> > & in,
        stream<ap_uint<PECount * AccType::width> > & out, const unsigned int numReps) {
    ap_uint<PECount * ActType::width> thin;
  ap_uint<PECount * AccType::width> accumulators[NumChannels/PECount];
#pragma HLS RESOURCE variable=accumulators core=RAM_2P_LUTRAM
        
    //call to thresholding library function
    for(unsigned int reps=0; reps<numReps; reps++){
        for(unsigned int pixel=0; pixel<ImgDim*ImgDim; pixel++){
      for(unsigned int fold=0; fold<NumChannels/PECount; fold++){
#pragma HLS PIPELINE II=1
        thin = in.read();
        ap_uint<PECount * AccType::width> accbank = accumulators[fold];
        for(unsigned int pe=0; pe<PECount; pe++){
        #pragma HLS UNROLL
          // Threshold and assign to right bits of output buffers
          unsigned int lowBit = pe * ActType::width;
          unsigned int highBit = (pe+1) * ActType::width - 1;
          ActType val = thin((pe+1) * ActType::width - 1,pe * ActType::width);
          AccType acc = accbank((pe+1) * AccType::width - 1,pe * AccType::width);
          AccType result;
          if(pixel == 0)
                  result = val;
          else
                  result = val+acc;
          accbank((pe+1) * AccType::width - 1,pe * AccType::width) = result;
        }
        accumulators[fold] = accbank;     
      }
        }
    for (unsigned int fold = 0; fold < NumChannels / PECount; fold++)
    {
      out.write(accumulators[fold]);
    }
    }
}



/**
 * \brief   LabelSelect_Batch - returns labels of top-5 in stream
 *
 * \tparam NumClasses   Number of classes of the dataset
 * \tparam PECount      Number of inputs to be processed in parallel
 * \tparam NumTop       Number of top classes to be selected in output
 * \tparam In_T         Datatype of the input
 * \tparam Out_T        Datatype of the output
 * 
 * \param in            Input stream
 * \param out           Output stream
 * \param numReps       Number of times the function has to be repeatedly executed (e.g. number of images)
 *
 */

template<
        // tensor size parameters
        unsigned int NumClasses,
        unsigned int PECount,
    unsigned int NumTop,
        typename In_T,
    typename Out_T>
void LabelSelect_Batch(stream<ap_uint<PECount * In_T::width> > & in,
        stream<Out_T> & out, const unsigned int numReps) { 

  const Out_T Out_T_MAX_VAL = (Out_T(-1)<0)? ~(1<<(Out_T::width-1)) : ~(0);
  CASSERT_DATAFLOW(Out_T_MAX_VAL >= NumClasses-1);

  const In_T In_T_MIN_VAL = (In_T(-1)<0)? 1<<(In_T::width-1) : 0;
  ap_uint<PECount * In_T::width> inval;

  Out_T toplabels[NumTop];
  #pragma HLS ARRAY_PARTITION variable=toplabels complete dim=1

  In_T topval[NumTop];
  #pragma HLS ARRAY_PARTITION variable=topval complete dim=1

  for(unsigned int reps=0; reps<numReps; reps++){
    unsigned int idx = 0;
    for(unsigned int topx=0; topx<NumTop; topx++){
      #pragma HLS UNROLL
      topval[topx] = In_T_MIN_VAL; 
    }
    for(unsigned int block=0; block<(NumClasses/PECount); block++){
      #pragma HLS PIPELINE II=1
      inval = in.read();
      for(unsigned int elem=0; elem<PECount; elem++){
        #pragma HLS UNROLL
        unsigned int lowBit = elem * In_T::width;
        unsigned int highBit = (elem+1) * In_T::width - 1;
        In_T val = inval(highBit,lowBit);
        for(unsigned int topx=0; topx<NumTop; topx++){
          #pragma HLS UNROLL
          if(val > topval[topx]){
            if(topx==(NumTop-1)){
              topval[topx] = val;
              toplabels[topx] = idx;
            } else if(val > topval[topx+1]){
              topval[topx] = topval[topx+1];
              toplabels[topx] = toplabels[topx+1];
            } else {
              topval[topx] = val;
              toplabels[topx] = idx;
            }
          }
        }
        idx++;
      }
    }

    for(unsigned int topx = 0; topx < NumTop; topx++){
      out.write(toplabels[NumTop - topx - 1]);
    }
  }
}


/**
 * \brief Pool_batch function
 *
 * The function performs a generic pool function (defined in pool.hpp) and works in conjuction 
 * with a sliding window unit performing im2col on the input data, allowing 
 * generic kernel and stride values
 *
 * \tparam Channels   Number of channels in the pool layer
 * \tparam PE         Number of channels in the pool layer computed in parallel
 * \tparam TotalK     Total kernel size of pooling (e.g. 3x3=9)
 * \tparam TSrcI      DataType of the input value (Slice)
 * \tparam TDstI      DataType of the output value (Slice)
 * \tparam TI         DataType of the input stream - safely deducible from the paramaters
 * \tparam TO         DataType of the output stream - safely deducible from the paramaters
 * \tparam TA         DataType of the function class (e.g. Max, Avg, Sum) - safely deducible from the paramaters
 *
 * \param in          Input stream
 * \param out         Output stream
 * \param function    Function class in the pool (Max, Avg, Sum)
 * \param reps        Number of time the function has to be repeatedly executed (e.g. number of images)
 */
template<
  unsigned Channels, unsigned PE, unsigned TotalK,
  typename TSrcI = Identity,typename TDstI = Identity,
  typename TI, typename TO, typename TA
>
void Pool_batch(hls::stream<TI> &in,
                  hls::stream<TO> &out,
                  TA  const &function,
                  int const  reps) {

  unsigned const  NF = Channels / PE;
  unsigned const  SF = TotalK;

  decltype(function.init())  accu[PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0
  unsigned  nf   = 0;
  unsigned  sf   = 0;
  unsigned const TOTAL_FOLD = NF * SF ;
  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelining the way we want
  for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++) {
#pragma HLS PIPELINE II=1
    TI  pixel_slice;
    pixel_slice = in.read();

    // Threshold Initialisation
    if(sf == 0) {
      for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
        accu[pe] = function.init();
      }
    }

    auto const  slice_channels = TSrcI()(pixel_slice,0);
    for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
        accu[pe] = function.pool(slice_channels(pe,0), accu[pe]);
    }

    // keep track of which folded synapse/neuron we are processing
    if(++sf == SF) {
      // produce output and clear accumulators
      auto  outElem = TDstI().template operator()<TO>();
      for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
          outElem(pe,0,1) = function.activate(accu[pe]); //
      }
      out.write(outElem);
      // next folded neuron or image
      sf = 0;
    }
  }
}

/**
 * \brief 2D max pool function function
 *
 * The function performs a max pool operation and works in conjuction 
 * with a sliding window unit performing im2col on the input data, allowing 
 * generic kernel and stride values. Also it supports input parallelism and arbitrary input and output widths
 * Modified from the original implementation by Xilinx.
 *
 * \tparam IFMChannels 		Number of Input Feature Maps
 * \tparam IFMDim 			  Width and Height of the Input Feature Map (assumed square)
 * \tparam PE             Number of channels in the pool layer computed in parallel
 * \tparam PoolDim        Dimension of the Max Pool kernel
 * \tparam Stride         Stride of the max pool kernel
 * \tparam TSrcI          DataType of the input value (Slice)
 * \tparam ActType        DataType of the input value, without Slice (e.g. ap_fixed<32,16>)
 * \tparam InStreamW      Width, in number of bits, of the input stream
 * \tparam OutStreamW     Width, in number of bits, of the output stream
 *
 * \param in              Input stream
 * \param out             Output stream
 * \param reps            Number of time the function has to be repeatedly executed (e.g. number of images)
 */
template<
  unsigned int IFMChannels,
  unsigned int IFMDim,
  unsigned int PE,
  unsigned int PoolDim,
  unsigned int Stride,
  typename TSrcI,
  typename ActType,
  int InStreamW,
  int OutStreamW
>
void Streaming_Maxpool_batch2d(hls::stream<ap_uint<InStreamW>> &in,hls::stream<ap_uint<OutStreamW>> &out,int const  reps) {
    #pragma HLS DATAFLOW

    const int FOLD = IFMChannels/PE;
    unsigned const InpPerImage = IFMDim * IFMDim * IFMChannels * TSrcI::width / InStreamW;
    hls::stream<ap_uint<PE*TSrcI::width> > width_adjusted_input("Streaming_Maxpool_batch2d.width_adjusted_input");
    hls::stream<ap_uint<PE*TSrcI::width> > input_generator_out("Streaming_Maxpool_batch2d.input_generator_out");
    hls::stream<ap_uint<PE*TSrcI::width> > pool_out("Streaming_Maxpool_batch2d.pool_out");
    auto const mem_r = ap_resource_dflt();

    StreamingDataWidthConverter_Batch<InStreamW, PE*TSrcI::width, InpPerImage>(in, width_adjusted_input, reps);
    
    ConvolutionInputGenerator_dws<PoolDim, IFMChannels,TSrcI::width, IFMDim, IFMDim/PoolDim, PE,Stride>(width_adjusted_input, input_generator_out, reps, mem_r);

    MaxPoolFunction<ActType,PoolDim> maxpool_fxn;
    Pool_batch<IFMChannels, PE, PoolDim*PoolDim,TSrcI, TSrcI>(input_generator_out,pool_out, maxpool_fxn, (IFMDim/PoolDim)*(IFMDim/PoolDim)*reps);

    StreamingDataWidthConverter_Batch<PE*TSrcI::width,OutStreamW, (IFMDim/PoolDim)*(IFMDim/PoolDim)*FOLD>(pool_out, out, reps);

}
#endif
