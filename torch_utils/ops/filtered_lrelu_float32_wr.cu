#include "filtered_lrelu.cu"
template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<float, true>(const filtered_lrelu_kernel_params& p);
