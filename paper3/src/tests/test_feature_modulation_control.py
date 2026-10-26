"""CPU unit checks for the exploratory feature-modulation comparator."""
import unittest
import torch
from paper3.core import TemporalFusion
from paper3.feature_modulation_control import FrozenFeatureModulation


class FeatureModulationTests(unittest.TestCase):
    def test_forward_masks_and_frozen_reference(self):
        torch.manual_seed(7)
        reference = TemporalFusion()
        reference.eval()
        model = FrozenFeatureModulation(reference.state_dict())
        model.eval()
        rgb = torch.randn(2, 16, 512)
        pose = torch.randn(2, 16, 13, 2)
        mask = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool)
        with torch.no_grad():
            output = model(rgb, pose, mask)
            baseline = reference(rgb * mask[:, 0, None, None],
                                 pose * mask[:, 1, None, None, None])
        self.assertEqual(tuple(output.shape), (2, 15))
        self.assertTrue(torch.allclose(output, baseline, atol=1e-5))
        self.assertTrue(all(not p.requires_grad for p in model.reference.parameters()))
        self.assertGreater(sum(p.numel() for p in model.parameters() if p.requires_grad), 0)

    def test_rejects_no_sensor(self):
        reference = TemporalFusion()
        model = FrozenFeatureModulation(reference.state_dict())
        with self.assertRaises(ValueError):
            model(torch.zeros(1,16,512),torch.zeros(1,16,13,2),torch.zeros(1,2,dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
