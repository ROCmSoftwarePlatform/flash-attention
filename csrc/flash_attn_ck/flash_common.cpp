/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash_common.hpp"

namespace flash {
int override_num_splits_if_necessary(int batch,
                                     int nhead,
                                     int max_seqlen_q,
                                     int hdim_q,
                                     int hdim_v,
                                     float p_drop,
                                     bool is_prefill,
                                     int num_splits)
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return num_splits;
    }

    hipDeviceProp_t props{};
    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return num_splits;
    }

    const int kM0 = [&] {
        // get kM0 for prefill phase
        if(is_prefill)
        {
            return 128;
        }

        // get kM0 for decode phase
        /// TODO: take dtype=fp8/bf8 into consideration
        const std::map<int, int> hdim_to_m0 = {
            {32, 32},
            {64, 64},
            // {96,  64},
            {128, 64},
            {256, 64},
        };

        for(auto [hdim, m0] : hdim_to_m0)
        {
            if(hdim_q <= hdim && hdim_v <= hdim)
            {
                return m0;
            }
        }

        return 64; // meet unsupported hdim_q/hdim_v
    }();
    // const int kN1 = hdim_v;

    const int num_m_blocks = (max_seqlen_q + kM0 - 1) / kM0;
    // const int num_n_blocks = (hdim_v + kN1 - 1) / kN1; // always 1

    if(num_splits < 1 && p_drop == 0.0f)
    {
        return num_splits_heuristic_ck(batch * nhead * num_m_blocks, props.multiProcessorCount * 2, 8);
    }

    return num_splits;
}

} // namespace flash
