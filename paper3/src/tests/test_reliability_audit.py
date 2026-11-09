import unittest
import pandas as pd
from paper3.reliability_audit import paired_quality_audit

class ReliabilityAuditTests(unittest.TestCase):
    def rows(self,reverse=False):
        rows=[]
        for vid in ("0001","0002","0003","0004"):
            i=int(vid)
            clean=.8+i*.01
            bad=.2+i*.01
            if reverse:clean,bad=bad,clean
            for state,reliability in (("clean",clean),("corrupt",bad)):
                rows.append(dict(seed=42,condition="blur",severity=.5,modality="rgb",
                    video_id=vid,state=state,reliability=reliability))
        return rows
    def test_correct_polarity(self):
        result=paired_quality_audit(pd.DataFrame(self.rows()),n_boot=10)
        self.assertEqual(result.iloc[0].auroc,1.)
        self.assertFalse(bool(result.iloc[0].polarity_warning))
    def test_inverted_polarity(self):
        result=paired_quality_audit(pd.DataFrame(self.rows(True)),n_boot=10)
        self.assertEqual(result.iloc[0].auroc,0.)
        self.assertTrue(bool(result.iloc[0].polarity_warning))
    def test_missing_pair(self):
        with self.assertRaises(ValueError):
            paired_quality_audit(pd.DataFrame(self.rows()[:-1]),n_boot=10)
    def test_duplicate(self):
        rows=self.rows()
        with self.assertRaises(ValueError):
            paired_quality_audit(pd.DataFrame(rows+[rows[0]]),n_boot=10)
if __name__=="__main__":unittest.main()
