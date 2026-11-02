

// ============================================================================
// SNN Inference from PS-written BRAM Features for Vitis HLS
// Resource-reduced + FC1 feature-parallel version
// ============================================================================

#include "ap_int.h"
#include "fpga_weights.h"

typedef ap_int<16> data_t;
typedef ap_int<16> weight_t;
typedef ap_int<32> acc_t;
typedef ap_uint<1> spike_t;

// ---------------------------------------------------------------------------
// LIF parameters
// ---------------------------------------------------------------------------
static const acc_t BETA_NUM = 115;
static const acc_t LIF_THRESHOLD = WEIGHT_SCALE;

// ---------------------------------------------------------------------------
// Unroll factors
// ---------------------------------------------------------------------------
static const int FC1_NEURON_UNROLL = 4;
static const int FC2_NEURON_UNROLL = 4;
static const int FC3_OUT_UNROLL    = 4;

// ============================================================================
// LIF update
// ============================================================================
static spike_t lif_update(acc_t input_current, acc_t &mem)
{
#pragma HLS INLINE

    // Determine reset from the previous membrane voltage.
    spike_t previous_spike = (mem > LIF_THRESHOLD);

    // Integrate current and apply the delayed reset.
    mem = ((mem * BETA_NUM) >> 7) + input_current;

    if (previous_spike) {
        mem -= LIF_THRESHOLD;
    }

    // Generate the current spike without immediately resetting mem.
    return (mem > LIF_THRESHOLD) ? spike_t(1) : spike_t(0);
}

// ============================================================================
// Read input features from BRAM
// ============================================================================
static void read_features_from_bram(
    volatile data_t *feature_mem,
    data_t features[T_STEPS][N_FEATURES]
) {
#pragma HLS INLINE off

    for (int i = 0; i < T_STEPS * N_FEATURES; i++) {
#pragma HLS PIPELINE II=1

        int t = i / N_FEATURES;
        int f = i % N_FEATURES;

        features[t][f] = feature_mem[i];
    }
}

// ============================================================================
// FC1 + LIF
// ============================================================================
static void fc1_lif(
    data_t x[N_FEATURES],
    spike_t spk1[HIDDEN1],
    acc_t mem1[HIDDEN1]
) {
#pragma HLS INLINE off

#pragma HLS ARRAY_PARTITION variable=fc1_w cyclic factor=FC1_NEURON_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=fc1_w cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=fc1_b cyclic factor=FC1_NEURON_UNROLL dim=1

#pragma HLS ARRAY_PARTITION variable=x cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=mem1 cyclic factor=FC1_NEURON_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=spk1 cyclic factor=FC1_NEURON_UNROLL dim=1

    for (int h = 0; h < HIDDEN1; h++) {
#pragma HLS UNROLL factor=FC1_NEURON_UNROLL

        acc_t acc = fc1_b[h];

        for (int i = 0; i < N_FEATURES; i += 2) {
#pragma HLS PIPELINE II=1

            if (x[i] != 0) {
                acc += ((acc_t)x[i] * (acc_t)fc1_w[h][i]) >> 7;
            }

            if (x[i + 1] != 0) {
                acc += ((acc_t)x[i + 1] * (acc_t)fc1_w[h][i + 1]) >> 7;
            }
        }

        spk1[h] = lif_update(acc, mem1[h]);
    }
}

// ============================================================================
// FC2 + LIF
// ============================================================================
static void fc2_lif(
    spike_t spk1[HIDDEN1],
    spike_t spk2[HIDDEN2],
    acc_t mem2[HIDDEN2]
) {
#pragma HLS INLINE off

#pragma HLS ARRAY_PARTITION variable=fc2_w cyclic factor=FC2_NEURON_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=fc2_b cyclic factor=FC2_NEURON_UNROLL dim=1

#pragma HLS ARRAY_PARTITION variable=mem2 cyclic factor=FC2_NEURON_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=spk2 cyclic factor=FC2_NEURON_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=spk1 cyclic factor=FC1_NEURON_UNROLL dim=1

    for (int h = 0; h < HIDDEN2; h++) {
#pragma HLS UNROLL factor=FC2_NEURON_UNROLL

        acc_t acc = fc2_b[h];

        for (int i = 0; i < HIDDEN1; i++) {
#pragma HLS PIPELINE II=1

            if (spk1[i]) {
                acc += fc2_w[h][i];
            }
        }

        spk2[h] = lif_update(acc, mem2[h]);
    }
}

// ============================================================================
// FC3 output layer
// ============================================================================
static void fc3_lif(
    spike_t spk2[HIDDEN2],
    spike_t spk3[NUM_CLASSES],
    acc_t mem3[NUM_CLASSES]
) {
#pragma HLS INLINE off

#pragma HLS ARRAY_PARTITION variable=fc3_w complete dim=1
#pragma HLS ARRAY_PARTITION variable=fc3_b complete dim=1
#pragma HLS ARRAY_PARTITION variable=spk3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=mem3 complete dim=1

    for (int o = 0; o < NUM_CLASSES; o++) {
#pragma HLS UNROLL factor=FC3_OUT_UNROLL

        acc_t acc = fc3_b[o];

        for (int i = 0; i < HIDDEN2; i++) {
#pragma HLS PIPELINE II=1
            if (spk2[i]) {
                acc += fc3_w[o][i];
            }
        }

        spk3[o] = lif_update(acc, mem3[o]);
    }
}

// ============================================================================
// Argmax
// ============================================================================
static int argmax_logits(acc_t logits_sum[NUM_CLASSES])
{
#pragma HLS INLINE

    int best = 0;
    acc_t best_val = logits_sum[0];

    for (int i = 1; i < NUM_CLASSES; i++) {
#pragma HLS UNROLL

        if (logits_sum[i] > best_val) {
            best_val = logits_sum[i];
            best = i;
        }
    }

    return best;
}

// ============================================================================
// SNN core
// ============================================================================
static int snn_core(
    data_t features[T_STEPS][N_FEATURES],
    acc_t logits_out[NUM_CLASSES]
)
{
#pragma HLS INLINE off

    acc_t mem1[HIDDEN1];
    acc_t mem2[HIDDEN2];
    acc_t mem3[NUM_CLASSES];

    spike_t spk1[HIDDEN1];
    spike_t spk2[HIDDEN2];
    spike_t spk3[NUM_CLASSES];

    acc_t logits_sum[NUM_CLASSES];

#pragma HLS ARRAY_PARTITION variable=mem1 cyclic factor=FC1_NEURON_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=mem2 cyclic factor=FC2_NEURON_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=mem3 complete dim=1

#pragma HLS ARRAY_PARTITION variable=spk1 cyclic factor=FC1_NEURON_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=spk2 cyclic factor=FC2_NEURON_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=spk3 complete dim=1

#pragma HLS ARRAY_PARTITION variable=logits_sum complete dim=1

    for (int i = 0; i < HIDDEN1; i++) {
#pragma HLS PIPELINE II=1
        mem1[i] = 0;
        spk1[i] = 0;
    }

    for (int i = 0; i < HIDDEN2; i++) {
#pragma HLS PIPELINE II=1
        mem2[i] = 0;
        spk2[i] = 0;
    }

    for (int i = 0; i < NUM_CLASSES; i++) {
#pragma HLS UNROLL
        mem3[i] = 0;
        spk3[i] = 0;
        logits_sum[i] = 0;
    }

    for (int t = 0; t < T_STEPS; t++) {
        fc1_lif(features[t], spk1, mem1);
        fc2_lif(spk1, spk2, mem2);
        fc3_lif(spk2, spk3, mem3);

        for (int o = 0; o < NUM_CLASSES; o++) {
#pragma HLS UNROLL
            logits_sum[o] += mem3[o];
        }
    }

    for (int o = 0; o < NUM_CLASSES; o++) {
    #pragma HLS UNROLL
        logits_out[o] = logits_sum[o];
    }

    return argmax_logits(logits_sum);
}


// ============================================================================
// Top-level HLS function
// PS writes features into BRAM.
// HLS reads BRAM and returns prediction through AXI-Lite.
// ============================================================================
void snn_mlp(
    volatile data_t *feature_mem,
    volatile ap_uint<8> *prediction,
    volatile acc_t logits_out[NUM_CLASSES]
) {
#pragma HLS INTERFACE m_axi port=feature_mem offset=slave bundle=gmem depth=384
#pragma HLS INTERFACE s_axilite port=feature_mem bundle=CTRL
#pragma HLS INTERFACE s_axilite port=prediction bundle=CTRL
#pragma HLS INTERFACE s_axilite port=logits_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    data_t features[T_STEPS][N_FEATURES];

#pragma HLS BIND_STORAGE variable=features type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=features complete dim=1

    read_features_from_bram(feature_mem, features);

    acc_t local_logits[NUM_CLASSES];
    #pragma HLS ARRAY_PARTITION variable=local_logits complete dim=1

    int pred = snn_core(features, local_logits);

    *prediction = (ap_uint<8>)pred;

    for (int i = 0; i < NUM_CLASSES; i++) {
    #pragma HLS UNROLL
        logits_out[i] = local_logits[i];
    }
}
