import torch
import torchmetrics
from metrics import ECE, ACE,  BrierScore, Reliability, Refinement
from torchmetrics.classification import BinaryCalibrationError
import unittest
import numpy as np

metric_ECE = ECE(bins=10)
metric_ACE = ACE(bins=10)
metric_ECE_torch = BinaryCalibrationError(n_bins = 10)
metric_brier = BrierScore()
metric_reliability = Reliability(bins=10)
metric_refinement = Refinement(bins=10)

# Reseting internal state such that metric ready for new data
metric_ECE.reset()
metric_ACE.reset()
metric_ECE_torch.reset()
metric_brier.reset()
metric_reliability.reset()
metric_refinement.reset()

y_class_case1=torch.tensor([1, 0, 1, 1, 0])
y_hat_case1=torch.tensor([0.12, 0.41, 0.61, 0.71, 0.94])
y_class_case2=torch.tensor([0,1,0,0,1,0,1,1,1,1,0,0])
y_hat_case2=torch.tensor([0.01, 0.04, 0.21, 0.46, 0.46, 0.58, 0.61, 0.77, 0.86, 0.87, 0.89, 0.94])
y_class_case3=torch.tensor([0, 1, 0, 1])
y_hat_case3=torch.tensor([0.41, 0.56, 0.67, 0.90])


y_class_case4=torch.tensor([0, 1,    0,   1,   0,  0,    0,   1,   1,   1,  1,   0,   1])
y_hat_case4=torch.tensor([0.0, 0.0, 0.0, 0.0,  0.2, 0.2,0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])



y_class_prefect=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
y_hat_perfect=torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])


y_class_compare = torch.ones(1000)
y_class_compare[:500] = 0

y_hat_compare = y_class_compare.clone()
y_hat_compare[0:399] = 0
y_hat_compare[600:999] = 1
y_hat_compare[400:600] = 0.2


def calcCalibrationErrors(preds,target):
    n_batches = 1
    for i in range(n_batches):
        # metric on current batch
        metric_ECE(preds, target)
        metric_ACE(preds, target)
        metric_ECE_torch(preds, target)
        metric_brier(preds, target)
        metric_reliability(preds, target)
        metric_refinement(preds, target)

    # metric on all batches using custom accumulation
    ece_total = metric_ECE.compute()
    ace_total = metric_ACE.compute()
    ece_torch_total = metric_ECE_torch.compute()
    brier_total = metric_brier.compute()
    refinement_total = metric_refinement.compute()
    reliability_total = metric_reliability.compute()
    metric_ECE.reset()
    metric_ACE.reset()
    metric_ECE_torch.reset()
    metric_brier.reset()
    metric_reliability.reset()
    metric_refinement.reset()
    return ece_total, ace_total, ece_torch_total, brier_total, refinement_total, reliability_total

class TestCalculateErrors(unittest.TestCase):
    
    def test_ECE_case1(self):
        test_true_case1 = y_class_case1
        test_score_case1 = y_hat_case1
        result_case1 = calcCalibrationErrors(test_score_case1, test_true_case1)
        self.assertAlmostEqual(result_case1[0].item(), (2.91/5))
        # Reseting internal state such that metric ready for new data
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_ECE_case2(self):
        test_true_case2 = y_class_case2
        test_score_case2 = y_hat_case2
        result_case2 = calcCalibrationErrors(test_score_case2, test_true_case2)
        self.assertAlmostEqual(result_case2[0].item(), (4/12))
        # Reseting internal state such that metric ready for new data
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
    metric_reliability.reset()
    metric_refinement.reset()

    def test_ECE_case3(self):
        test_true_case3 = y_class_case3
        test_score_case3 = y_hat_case3
        result_case3 = calcCalibrationErrors(test_score_case3, test_true_case3)
        self.assertAlmostEqual(result_case3[0].item(), (0.405))
        # Reseting internal state such that metric ready for new data
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_ECE_case4(self):
        test_true_case4 = y_class_case4
        test_score_case4 = y_hat_case4
        result_case4 = calcCalibrationErrors(test_score_case4, test_true_case4)
        self.assertAlmostEqual(torch.round(result_case4[0], decimals = 4).item(), (0.2769))
        # Reseting internal state such that metric ready for new data
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_ECE_perfect(self):
        test_true_perfect=y_class_prefect
        test_score_perfect=y_hat_perfect
        result_perfect = calcCalibrationErrors(test_score_perfect, test_true_perfect)
        self.assertAlmostEqual(result_perfect[0].item(), (0.0))
        # Reseting internal state such that metric ready for new data
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_ACE_case1(self):
        test_true_case1 = y_class_case1
        test_score_case1 = y_hat_case1
        result_case1 = calcCalibrationErrors(test_score_case1, test_true_case1)
        self.assertAlmostEqual(result_case1[1].item(), (2.91/5))
        # Reseting internal state such that metric ready for new data
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_ACE_case2(self):
        test_true_case2 = y_class_case2
        test_score_case2 = y_hat_case2
        result_case2 = calcCalibrationErrors(test_score_case2, test_true_case2)
        self.assertAlmostEqual(result_case2[1].item(), (5.46/12))
        # Reseting internal state such that metric ready for new data
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_ACE_case3(self):
        test_true_case3 = y_class_case3
        test_score_case3 = y_hat_case3
        result_case3 = calcCalibrationErrors(test_score_case3, test_true_case3)
        self.assertAlmostEqual(result_case3[1].item(), (0.405))
        # Reseting internal state such that metric ready for new data
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()
        
    def test_ACE_case4(self):
        test_true_case4 = y_class_case4
        test_score_case4 = y_hat_case4
        result_case4 = calcCalibrationErrors(test_score_case4, test_true_case4)
        self.assertAlmostEqual(torch.round(result_case4[1], decimals = 4).item(), (0.2769))
        # Reseting internal state such that metric ready for new data
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_torchmetrics_ECE_1(self):
        test_true_torch_case1 = y_class_case1
        test_score_torch_case1 = y_hat_case1
        result_torch_case1 = calcCalibrationErrors(test_score_torch_case1, test_true_torch_case1)
        self.assertAlmostEqual(torch.round(result_torch_case1[2], decimals=3).item(), (2.91/5))
        # Reseting internal state such that metric ready for new data
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_torchmetrics_ECE_2(self):
        test_true_torch_case2 = y_class_case2
        test_score_torch_case2 = y_hat_case2
        result_torch_case2 = calcCalibrationErrors(test_score_torch_case2, test_true_torch_case2)
        self.assertAlmostEqual(result_torch_case2[2].item(), (4/12))
        # Reseting internal state such that metric ready for new data
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_torchmetrics_ECE_3(self):
        test_true_torch_case3 = y_class_case3
        test_score_torch_case3 = y_hat_case3
        result_torch_case3 = calcCalibrationErrors(test_score_torch_case3, test_true_torch_case3)
        self.assertAlmostEqual(result_torch_case3[2].item(), (0.405))
        # Reseting internal state such that metric ready for new data
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_torchmetrics_ECE_4(self):
        test_true_torch_case4 = y_class_case4
        test_score_torch_case4 = y_hat_case4
        result_torch_case4 = calcCalibrationErrors(test_score_torch_case4, test_true_torch_case4)
        self.assertAlmostEqual(torch.round(result_torch_case4[2], decimals = 4).item(), (0.2769))
        # Reseting internal state such that metric ready for new data
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_torchmetrics_ECE_perfect(self):
        test_true_perfect=y_class_prefect
        test_score_perfect=y_hat_perfect
        result_perfect = calcCalibrationErrors(test_score_perfect, test_true_perfect)
        self.assertAlmostEqual(result_perfect[2].item(), (0.0))
        # Reseting internal state such that metric ready for new data
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_brier_score_case1(self):
        test_true_case1 = y_class_case1
        test_score_case1 = y_hat_case1
        result_brier_case1 = calcCalibrationErrors(test_score_case1, test_true_case1)
        self.assertAlmostEqual(result_brier_case1[3].item(), (0.41246))
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_brier_score_case2(self):
        test_true_case2 = y_class_case2
        test_score_case2 = y_hat_case2
        result_brier_case2 = calcCalibrationErrors(test_score_case2, test_true_case2)
        self.assertAlmostEqual(result_brier_case2[3].item(), (3.7226/12))
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_compare_ECEs(self):
        result_ECE_r= calcCalibrationErrors(y_hat_compare, y_class_compare)[0]
        result_ECE_torch = calcCalibrationErrors(y_hat_compare, y_class_compare)[2]
        self.assertAlmostEqual(torch.round(result_ECE_r).item(), torch.round(result_ECE_torch).item())
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()
        
    def test_reliability_refinement_1(self):
        test_true_case1 = y_class_case1
        test_score_case1 = y_hat_case1
        brier1, ref1, rel1= calcCalibrationErrors(test_score_case1, test_true_case1)[3:]
        self.assertEqual((ref1+rel1), brier1)
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_reliability_refinement_2(self):
        test_true_case2 = y_class_case2
        test_score_case2 = y_hat_case2
        brier2, ref2, rel2= calcCalibrationErrors(test_score_case2, test_true_case2)[3:]
        self.assertEqual((ref2+rel2), brier2)
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_reliability_refinement_3(self):
        test_true_case3 = y_class_case3
        test_score_case3 = y_hat_case3
        brier3, ref3, rel3= calcCalibrationErrors(test_score_case3, test_true_case3)[3:]
        self.assertEqual((ref3+rel3), brier3)
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_reliability_refinement_4(self):
        test_true_case4 = y_class_case4
        test_score_case4 = y_hat_case4
        brier4, ref4, rel4= calcCalibrationErrors(test_score_case4, test_true_case4)[3:]
        self.assertEqual((ref4+rel4), brier4)
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_reliability_refinement_perfect(self):
        test_true_prefect = y_class_prefect
        test_score_prefect = y_hat_perfect
        brier_prefect, ref_prefect, rel_prefect= calcCalibrationErrors(test_score_prefect, test_true_prefect)[3:]
        self.assertEqual((ref_prefect+rel_prefect), brier_prefect)
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()

    def test_reliability_compare(self):
        test_true_compare = y_class_compare
        test_score_compare = y_hat_compare
        brier_compare, ref_compare, rel_compare= calcCalibrationErrors(test_score_compare, test_true_compare)[3:]
        self.assertEqual((ref_compare+rel_compare), brier_compare)
        metric_ECE.reset()
        metric_ACE.reset()
        metric_ECE_torch.reset()
        metric_brier.reset()
        metric_reliability.reset()
        metric_refinement.reset()
    
if __name__ == '__main__':
    unittest.main()