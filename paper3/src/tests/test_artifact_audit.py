import unittest
import pandas as pd
from paper3.artifact_audit import audit_predictions

class ArtifactAuditTests(unittest.TestCase):
    def setUp(self):
        self.rows=[]
        for variant in ("v2","specialist_rule"):
            for video in ("0001","0002"):
                self.rows.append(dict(seed=42,condition="clean",severity=0.,variant=variant,
                    video_id=video,class_id=int(video)-1,prediction=int(video)-1,
                    correct=1,confidence=.9))
    def test_valid(self):
        self.assertTrue(audit_predictions(pd.DataFrame(self.rows),2)["valid"])
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):audit_predictions(pd.DataFrame(self.rows+[self.rows[0]]),2)
    def test_label_mismatch_rejected(self):
        rows=[dict(r) for r in self.rows]
        rows[-1]["class_id"]=0
        rows[-1]["prediction"]=0
        with self.assertRaises(ValueError):audit_predictions(pd.DataFrame(rows),2)
    def test_incomplete_rejected(self):
        with self.assertRaises(ValueError):audit_predictions(pd.DataFrame(self.rows[:-1]),2)
    def test_wrong_correctness_rejected(self):
        rows=[dict(r) for r in self.rows]
        rows[0]["correct"]=0
        with self.assertRaises(ValueError):audit_predictions(pd.DataFrame(rows),2)
if __name__=="__main__":unittest.main()
