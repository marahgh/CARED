# When to Act and When to Ask: Policy Learning With Deferral Under Hidden Confounding

A method for learning policies from observational data, where the
policy model can predict a treatment assignment or defer to an expert under hidden confounding.

[//]: # (Replication code for []&#40;&#41;. )

## Requirements

* [econml](https://github.com/microsoft/EconML)
* [xgboost](https://pypi.org/project/xgboost/)
* [doubleml](https://github.com/DoubleML/doubleml-for-py)
* [sklearn-quantile](https://pypi.org/project/sklearn-quantile/)
* [pytorch](https://pytorch.org/)
* [ray](https://pypi.org/project/ray/)
* [pytorch-lightning](https://www.pytorchlightning.ai/)

## Usage Example
We train a `CARED` policy on observational train and validation sets $Z=(X, A, Y)$: `ds_train` and `ds_test`, predict the policy on a test set `ds_test`.
Given the `CAPO_bounds` We get from the `BLearner` with a specified confounding degree of `gamma`. We then train the policy as follows: 
```Python
import torch
import torch.nn as nn
from models.lce_policy.lce_policy import LCE_Policy

# Logistic Policy learner
policy_model = nn.Sequential(
    nn.Linear(features_num, treatments_num + 1),
    nn.Sigmoid())
# Policy model that optimizes the $L_{CE}$ Objective
lce_policy = LCE_Policy(tau_hat=CAPO_bounds,
	policy_model=policy_model,
	use_rho=True,                              
	gamma=gamma,                                
	higher_better=True)

# Train the policy model
lce_policy.fit(ds_train=ds_train, ds_valid=ds_valid, devices=devices)
# Predict on the test set
lce_pi_with_deferral = lce_policy.predict(ds_test=ds_test)
# Replace all deferrals with the expert's actions to get the final treatment assignment
lce_pi, deferral_count, deferral_rate = update_expert_preds(preds=lce_logistic_pi,
	expert_labels=torch.Tensor(
	ds_test.t).type(torch.LongTensor))

```
## Replication Code for Paper

The following commands will replicate the figures from the []() paper.

* For Figure 1, run `synthetic_experiment.py`
* For Figure 2, run `ihdp_experiment.py`

