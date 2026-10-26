"""Synthetic, CPU-only integration tests for the one-shot novelty campaign."""
import unittest
import torch
from paper3.core import TemporalFusion
from paper3.feature_modulation_control import FrozenFeatureModulation
from paper3.minimal_novelty import fit_film, predict
from paper3.final_campaign import train_mask, MaskAwareReference


class MinimalNoveltyIntegration(unittest.TestCase):
    def test_one_epoch_three_masks_and_predictions(self):
        torch.manual_seed(11)
        base=TemporalFusion()
        base.eval()
        r=torch.randn(9,16,512)
        p=torch.randn(9,16,13,2)
        y=torch.arange(9)%3
        train=(r[:6],p[:6],y[:6])
        val=(r[6:],p[6:],y[6:])
        film_ck,hist=fit_film(base.state_dict(),train,val,42,'cpu',epochs=1)
        self.assertEqual(len(hist),1)
        film=FrozenFeatureModulation(base.state_dict())
        film.load_state_dict(film_ck['model'])
        film.eval()
        mask_ck,_=train_mask(base.state_dict(),train,val,42,'cpu',epochs=1)
        masknet=MaskAwareReference(base.state_dict())
        masknet.load_state_dict(mask_ck['model'])
        masknet.eval()
        for mask in ((1,1),(0,1),(1,0)):
            for kind,model in (('film',film),('mask',masknet),('reference',base)):
                logits=predict(model,base,*val[:2],mask,kind,'cpu')
                self.assertEqual(tuple(logits.shape),(3,15))
                self.assertTrue(torch.isfinite(logits).all())


if __name__=='__main__':
    unittest.main()
