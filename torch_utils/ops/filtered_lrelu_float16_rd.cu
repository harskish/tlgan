#include "filtered_lrelu.cu"
template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<c10::Half, false>(const filtered_lrelu_kernel_params& p);
