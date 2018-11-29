# Caffe
 The caffe environment is from https://github.com/Tongcheng/caffe
  
 
# Implement adjustable gradient clipping 
modify sgd_solver.cpp in your_caffe_root/src/caffe/solvers/, where we add the following codes in funciton ClipGradients():

Dtype rate = GetLearningRate();

const Dtype clip_gradients = this->param_.clip_gradients()/rate;


# Training:
   1. Preparing training/validation data using the files: generate_trainingset_x234/generate_testingset_x234 in "data" folder. "Train_291" folder contains 291 training images and "Set5" folder is a popular benchmark dataset.
   2. To train the model MODN_B4U25,  sh train_MODN_B4U25.sh

   
# Testing:
  1. we deploy the models with matcaffe, compile "make matcaffe" before testing
  2. We release the pretrained models:  MODN_B4U25 in "model" folder. To do testing on benchmark Set5. E.g., run file ./test/Densenet_sr_B4U25, the results are stored in "deply" folder, with both reconstructed images and PSNR/SSIM/IFCs in "test/B4U25.txt".

# Tips:
  The DenseBlock_layer proposed by Tongcheng didn't release allocated gpu memory during testing time, so the big size images can't be deployed in one matlab running time. We free the GPU allocation in the testing time in our proposed DenseBlock_layer.cu. To use DenseBlock_layer.cu, replace the "DenseBlock_layer.cu" in the "src/layers"  and "make all" (Noted only use this "DenseBlock_layer.cu" only works in testing time)
