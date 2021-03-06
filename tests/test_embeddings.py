import numpy as np

import mutation_prediction.embeddings.aaindex as aaindex
from mutation_prediction.data import Dataset, acid_to_index
from mutation_prediction.embeddings.acids import VHSE, AcidsOneHot, ProtVec, ZScales
from mutation_prediction.embeddings.other import MutInd


def test_acids_one_hot():
    sequence = np.asarray([0, 1, 2])
    positions = np.asarray([[0], [1]])
    acids = np.asarray([[2], [0]])
    num_mutations = np.asarray([1, 1])
    dataset = Dataset(sequence, positions, acids, num_mutations)
    x = AcidsOneHot().embed_update(dataset)
    assert np.all(
        x
        == np.asarray(
            [
                [
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ]
        )
    )


def test_z_scales():
    sequence = np.asarray([0, 5, 19])
    positions = acids = np.zeros((1, 0))
    num_mutations = np.asarray([0])
    dataset = Dataset(sequence, positions, acids, num_mutations)
    assert np.all(
        ZScales().embed_update(dataset)
        == np.asarray(
            [
                [
                    [0.24, -2.32, 0.60, -0.14, 1.30],
                    [2.05, -4.06, 0.36, -0.82, -0.38],
                    [-2.54, 2.44, 0.43, 0.04, -1.47],
                ]
            ]
        )
    )


def test_vhse():
    sequence = np.asarray([acid_to_index("A"), acid_to_index("K"), acid_to_index("V")])
    positions = acids = np.zeros((1, 0))
    num_mutations = np.asarray([0])
    dataset = Dataset(sequence, positions, acids, num_mutations)
    assert np.all(
        VHSE().embed_update(dataset)
        == np.asarray(
            [
                [
                    [0.15, -1.11, -1.35, -0.92, 0.02, -0.91, 0.36, -0.48],
                    [-1.17, 0.7, 0.7, 0.8, 1.64, 0.67, 1.63, 0.13],
                    [0.76, -0.92, -0.17, -1.91, 0.22, -1.4, -0.24, -0.03],
                ]
            ]
        )
    )


def test_read_aaindex1():
    matrix = aaindex.read_aaindex1()
    assert matrix.shape == (20, 553)
    assert np.allclose(
        matrix[:, 0],
        np.asarray(
            [
                4.35,
                4.65,
                4.76,
                4.29,
                4.66,
                3.97,
                4.63,
                3.95,
                4.36,
                4.17,
                4.52,
                4.75,
                4.44,
                4.37,
                4.38,
                4.5,
                4.35,
                3.95,
                4.7,
                4.6,
            ]
        ),
    )
    assert np.allclose(
        matrix[:, -1],
        np.asarray(
            [
                0,
                6,
                2.969,
                1.822,
                2.026,
                0,
                1.605,
                3.373,
                1.372,
                3.113,
                2.656,
                3,
                12,
                1.849,
                4.2,
                6,
                6,
                6,
                2.044,
                1.599,
            ]
        ),
    )


def test_read_aaindex3():
    matrix = aaindex.read_aaindex3()
    assert np.allclose(
        matrix[0, :, 0],
        np.asarray(
            [
                -2.6,
                -4.2,
                -2.8,
                -3.0,
                -5.1,
                -3.8,
                -4.0,
                -5.9,
                -3.1,
                -4.8,
                -4.6,
                -3.1,
                -3.4,
                -3.5,
                -3.4,
                -2.9,
                -3.3,
                -4.3,
                -5.2,
                -4.7,
            ]
        ),
    )
    assert np.allclose(
        matrix[17, :, -1],
        np.asarray(
            [
                -0.18,
                -1.22,
                0.59,
                0.74,
                -0.77,
                0.04,
                0.09,
                -0.9,
                0.81,
                -0.93,
                -0.31,
                0.54,
                0.17,
                0.23,
                0.43,
                0.2,
                -0.1,
                -0.31,
                -0.57,
                -0.38,
            ]
        ),
    )


def test_protvec():
    sequence = np.asarray(
        [
            acid_to_index("A"),
            acid_to_index("L"),
            acid_to_index("A"),
            acid_to_index("A"),
            acid_to_index("A"),
        ]
    )
    positions = np.asarray([[0, 1], [3, 0]])
    acids = np.asarray([[acid_to_index("L"), acid_to_index("A")], [acid_to_index("L"), 0]])
    num_mutations = np.asarray([2, 1])
    dataset = Dataset(sequence, positions, acids, num_mutations)

    # fmt: off
    laa = np.asarray([-0.137546, -0.135425, 0.121566, -0.038295, -0.212129, 0.040009, 0.078545, 0.029837, 0.138343, 0.049377, 0.025048, -0.050824, 0.058681, 0.086782, -0.073994, -0.024942, -0.099726, 0.024913, 0.048674, 0.006191, 0.015447, -0.029424, 0.08341, 0.012713, -0.026227, -0.123598, 0.021424, 0.068988, -0.058738, -0.049295, -0.023567, -0.083769, -0.049185, -0.127975, 0.151709, -0.108231, -0.049147, -0.039074, 0.019889, 0.103406, 0.017171, -0.046425, 0.026756, 0.024482, 0.011911, 0.089955, -0.051382, -0.046249, -0.080505, -0.059127, -0.03726, -0.035306, -0.094678, -0.016989, 0.025153, 0.010931, -0.098882, -0.033411, -0.080273, -0.005884, -0.018289, 0.211769, -0.11639, 0.196089, -0.09902, 0.030546, 0.064562, 0.081074, -0.010765, 0.026536, -0.080675, 0.017583, 0.115859, -0.032546, -0.080149, 0.099125, -0.017189, 0.009218, 0.095391, 0.011473, 0.043502, -0.013087, 0.085062, 0.088393, 0.064209, 0.017879, -0.066337, 0.114686, -0.032628, -0.103722, 0.133947, -0.156484, -0.048541, 0.141848, 0.081842, 0.070573, 0.006927, 0.035281, -0.138971, 0.105997])
    aaa = np.asarray([-0.17406, -0.095756, 0.059515, 0.039673, -0.375934, -0.115415, 0.090725, 0.173422, 0.29252, 0.190375, 0.094091, -0.197482, -0.135202, 0.075521, 0.110771, 0.047909, -0.391934, 0.073548, 0.103868, -0.045924, -0.009534, 0.055659, -0.000308, 0.215941, 0.084476, 0.061573, 0.128139, 0.184247, -0.100091, -0.126661, -0.005728, -0.038272, 0.180597, -0.15531, 0.056232, -0.005925, -0.085381, -0.056921, -0.04552, 0.265116, 0.090221, -0.209879, 0.205381, 0.023679, -0.092939, 0.072767, -0.105107, 0.011112, -0.160518, 0.042627, 0.15123, -0.162708, -0.083479, -0.146657, 0.091332, 0.109579, -0.101678, 0.091198, 0.005512, 0.047318, 0.078108, 0.203824, -0.100126, 0.294703, -0.158841, 0.029333, 0.078265, 0.018524, 0.117082, 0.212755, -0.171555, 0.029421, 0.149264, 0.046599, -0.184111, 0.294123, -0.101497, -0.030123, -0.009826, 0.007835, -0.106508, -0.166202, -0.024748, -0.090856, 0.056977, 0.047644, 0.018618, -0.034376, 0.087013, -0.278817, 0.244482, 0.015974, 0.012903, 0.137528, 0.13814, 0.005474, 0.070719, -0.164084, -0.179274, 0.184899])
    ala = np.asarray([-0.114085, -0.093288, 0.1558, -0.037351, -0.121446, 0.084037, 0.023819, 0.093442, 0.143256, 0.044627, -0.105535, -0.087031, -0.147241, 0.012367, 0.002243, -0.041897, -0.182035, 0.080363, 0.135968, 0.032804, -0.032578, -0.081669, -0.053846, 0.067459, -0.011088, -0.126765, 0.107745, 0.006742, -0.044855, -0.077349, 0.054623, -0.059747, -0.091474, -0.132825, 0.066172, -0.045547, -0.092852, -0.02058, -0.016422, 0.082413, 0.033061, -0.04089, -0.007287, -0.087691, 0.052176, 0.118353, -0.024096, -0.105387, -0.099548, 0.013934, 0.029569, -0.201143, -0.067125, -0.165826, 0.051696, -0.087732, 0.018178, 0.031197, 0.042558, 0.050231, 0.014453, 0.145755, -0.008099, 0.176967, -0.062075, 0.084897, 0.082622, 0.053887, -0.013649, 0.024531, -0.118647, 0.020675, 0.07342, -0.042731, -0.010526, 0.056253, -0.057254, 0.002313, 0.061039, 0.063857, 0.097188, -0.012246, -0.004977, 0.088645, 0.098434, -0.144351, -0.096486, 0.096045, -0.052779, -0.021758, 0.075584, -0.139661, 0.034863, 0.056078, 0.028975, -0.012233, 0.059669, 0.037811, -0.172493, 0.074655])
    lal = np.asarray([-0.188611, -0.002185, 0.108836, -0.126098, -0.001931, -0.017215, -0.056647, 0.043682, 0.039895, -0.084752, -0.080592, 0.012292, -0.030957, 0.045595, -0.066493, 0.069233, -0.034512, 0.003164, 0.038675, 0.086966, -0.030664, -0.011946, 0.043148, -0.008076, -0.039833, -0.189831, 0.030943, 0.099167, 0.05707, 0.011906, 0.110972, -0.155954, -0.033212, -0.085699, 0.018596, 0.000278, -0.111411, 0.135607, 0.07563, -0.017231, -0.072363, 0.061376, 0.004911, -0.13198, 0.069393, 0.149795, 0.03813, 0.029557, 0.056873, -0.130344, -0.188549, -0.097198, -0.024204, -0.113265, 0.002824, -0.091056, -0.093314, 0.048391, -0.045528, 0.098762, -0.049364, 0.102176, -0.029537, 0.127859, 0.039618, 0.000541, 0.093655, 0.033931, 0.019304, -0.069376, -0.102236, -0.037201, 0.11614, -0.147525, -0.059271, 0.07817, -0.1385, 0.007937, -0.045835, 0.009225, -0.099611, -0.099497, 0.073739, -0.033784, 0.125643, -0.030537, 0.023395, 0.089798, 0.060555, -0.08036, 0.142023, -0.155075, -0.020684, 0.017047, 0.0902, -0.026145, 0.027764, 0.055355, -0.16733, 0.201886])
    # fmt: on

    embedding = ProtVec().embed_update(dataset)
    assert np.allclose(embedding[0], (laa + aaa) / 2.0)
    assert np.allclose(embedding[1], (ala + lal) / 2.0)


def test_mut_ind():
    sequence = np.zeros((10,))
    positions = np.asarray([[0, 5], [0, 5], [0, 0], [0, 0], [5, 0]])
    acids = np.asarray([[2, 2], [1, 2], [2, 0], [1, 0], [2, 0]])
    num_mutations = np.asarray([2, 2, 1, 1, 1])
    dataset = Dataset(sequence, positions, acids, num_mutations)
    embedding = MutInd().embed_update(dataset)
    assert np.all(np.sum(embedding, axis=1) == num_mutations)
    assert np.all(embedding[0] == embedding[2] + embedding[4])
    assert np.all(embedding[1] == embedding[3] + embedding[4])
