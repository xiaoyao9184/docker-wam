from wrapt_timeout_decorator import *

from notebooks.inference_utils import (
    multiwm_dbscan
)

@timeout(60, use_signals=False)
def dbscan(bit_preds, mask_preds, epsilon, min_samples, **kwargs):
    print("multiwm task started.")
    return multiwm_dbscan(preds=bit_preds, masks=mask_preds, epsilon=epsilon, min_samples=min_samples)
