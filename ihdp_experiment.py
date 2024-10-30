from pathlib import Path
import json

import math
import random

import numpy as np
from scipy import stats

import torch
import torch.nn as nn
from joblib import Parallel, delayed

from models.crlogit.data_scenarios import *
import statsmodels.api as sm

from compute_blearner_bounds_ihdp import compute_all_intervals_BLearner, get_nuisances_models_RF, policy_value
from models.lce_policy.lce_policy import LCE_Policy, extract_results

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from datasets import IHDP

# ConfHAI imports
# from models.confhai.utils import *
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import Tensor


np.random.seed(42)
torch.manual_seed(42)
# Gamma dictionary for Learning the policies
GAMMAS = {
    "0.0": math.exp(0.0),
    "0.1": math.exp(0.1),
    "0.2": math.exp(0.2),
    "0.5": math.exp(0.5),
    "0.7": math.exp(0.7),
    "1.0": math.exp(1.0),
    "1.2": math.exp(1.2),
    "1.5": math.exp(1.5),
    "2.0": math.exp(2.0),
    "2.5": math.exp(2.5),
    "3.0": math.exp(3.0),
    "3.5": math.exp(3.5),
    "4.0": math.exp(4.0),
    # "4.5": math.exp(4.5),
    # "5.0": math.exp(5.0),
    # "5.5": math.exp(5.5),
    # "6.0": math.exp(6.0),
    # "6.5": math.exp(6.5),
    # "7.0": math.exp(7.0),
    # "7.5": math.exp(7.5),
    # "8.0": math.exp(8.0),
    # "8.5": math.exp(8.5),
    # "9.0": math.exp(9.0),
    # "9.5": math.exp(9.5),
    # "10.": math.exp(10.0),
}

# # Gammas dictionary for calculating the CAPO Bounds
# GAMMAS_B = {
#     "0.0": math.exp(0.0),
#     "0.1": math.exp(0.1),
#     "0.2": math.exp(0.2),
#     "0.5": math.exp(0.5),
#     "0.7": math.exp(0.7),
#     "1.0": math.exp(1.0),
#     "1.2": math.exp(1.2),
#     "1.5": math.exp(1.5),
#     "2.0": math.exp(2.0),
#     "2.5": math.exp(2.5),
#     "3.0": math.exp(3.0),
#     "3.5": math.exp(3.5),
#     "4.0": math.exp(4.0),
#     "4.5": math.exp(4.5),
#     "5.0": math.exp(5.0),
#     "5.5": math.exp(5.5),
#     "6.0": math.exp(6.0),
#     "6.5": math.exp(6.5),
#     "7.0": math.exp(7.0),
#     "7.5": math.exp(7.5),
#     "8.0": math.exp(8.0),
#     "8.5": math.exp(8.5),
#     "9.0": math.exp(9.0),
#     "9.5": math.exp(9.5),
#     "10.": math.exp(10.0),
# }

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 50

rc = {
    "figure.constrained_layout.use": True,
    "axes.titlesize": 20,
}
sns.set_theme(style="darkgrid", palette="colorblind", rc=None)

sns.set(style="whitegrid", palette="colorblind")
params = {
    "figure.constrained_layout.use": True,
    "axes.labelsize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
    "font.size": 18,
}
plt.rcParams.update(params)

_FUNCTION_COLOR = "#ad8bd6"


def ggplot_log_style_deferral(figsize, log_y=False, loc_maj_large=True):
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Give plot a gray background like ggplot.
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 16
    ax.set_facecolor('#EBEBEB')
    # Remove border around plot.
    [ax.spines[side].set_visible(False) for side in ax.spines]
    # Style the grid.
    ax.grid(which='major', color='white', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)

    return ax

# Functions needed to the CRLogit Method
def real_risk_prob(prob_1, x, u):
    n = len(u)
    prob_1 = np.asarray(prob_1)
    return prob_1 * real_risk_(np.ones(n), x, u) + (1 - prob_1) * real_risk_(np.zeros(n), x, u)


def real_risk_prob_ihdp(prob_1, Y1, Y0):
    prob_1 = np.asarray(prob_1)
    return prob_1 * Y1 + (1 - prob_1) * Y0

def ihdp_q0_baseline_p(x_train, t_train, x):
    propensity_model = LogisticRegression(C=1, penalty='elasticnet', solver='saga', l1_ratio=0.7, max_iter=10000)
    propensity_model.fit(X=x_train, y=t_train)
    q_0 = propensity_model.predict_proba(x)[:, [1]]

    return q_0


def default_policy(q_0, p):
    return q_0

# Parameters for CRLogit Method
N_REPS = 1
GAMS = np.fromiter(GAMMAS.keys(), dtype=float)
# GAMS_B = np.fromiter(GAMMAS_B.keys(), dtype=float)

ENN = 4000
d = 24  # dimension of x
th_ctrl = np.zeros(d + 1)
th_ctrl[-1] = -1000

save_params = {'stump': 'ihdp', 'exp_name': 'testing'}

# Calculate means and stds
def calc_means_sems(results_list):
    means = np.mean(np.asarray(results_list), axis=0)
    stds = stats.sem(np.asarray(results_list), axis=0)
    return np.asarray(means), np.asarray(stds)

# Update the tratments of the expert for deferrals
def update_expert_preds(preds, expert_labels):
    defer_count = 0
    # updated_preds = preds.detach().clone()
    updated_preds = torch.Tensor(preds)
    for j in range(len(preds)):
        if preds[j] == 2:
            defer_count = defer_count + 1
            updated_preds[j] = expert_labels[j]

    deferral_rate = (defer_count / len(preds))
    return updated_preds, defer_count, deferral_rate

# Update the IHDP dataset with the designed expert based on the given feature
def update_expert(ds, feature):
    tau_true = torch.tensor(ds.mu1 - ds.mu0)
    # Oracle policy
    oracle_pi = (tau_true > 0) * 1
    # Current expert
    current_policy = ds.t
    y_current_expert = ds.y
    cond_train = np.asarray(ds.df[feature])
    for i in range(len(current_policy)):
        if cond_train[i] == 1:
            current_policy[i] = oracle_pi[i]
            if oracle_pi[i] == 0:
                y_current_expert[i] = ds.y0[i]
            else:
                y_current_expert[i] = ds.y1[i]
    ds.t = current_policy
    ds.y = y_current_expert
    return ds


def get_expert_by_feature(ds, feature):
    # ds = update_expert(ds=ds, feature=feature)
    expert_t = ds.t
    expert_y = ds.y
    return expert_t, expert_y

# Saving policies in a json file
def save_policies(policies, file_path):
    tmp_policies = policies
    for k, v in tmp_policies.items():
        tmp_policies[k] = (v.numpy()).tolist()
    with file_path.open(mode="w") as fp:
        json.dump(tmp_policies, fp)

# Loading policies from a json file
def load_policies(file_path):
    with file_path.open(mode="r") as fp:
        policies = json.load(fp)
    for k, v in policies.items():
        policies[k] = torch.Tensor(v)
    return policies



# Calculate the true gamma for the IHDP dataset
def get_ihdp_true_gamma(ihdp_train_ds, ihdp_test_ds):
    clf = LogisticRegression(C=1, penalty='elasticnet', solver='saga', l1_ratio=0.7, max_iter=10000)
    x = ihdp_train_ds.x
    x_test = ihdp_test_ds.x
    clf.fit(x, ihdp_train_ds.t)
    nominal_propensity_test = clf.predict_proba(x_test)[:, [1]]
    clf = LogisticRegression(C=1, penalty='elasticnet', solver='saga', l1_ratio=0.7, max_iter=10000)
    x_full = np.concatenate((ihdp_train_ds.x, ihdp_train_ds.u), axis=1)
    x_full_test = np.concatenate((ihdp_test_ds.x, ihdp_test_ds.u), axis=1)
    clf.fit(x_full, ihdp_train_ds.t)
    full_propensity_test = clf.predict_proba(x_full_test)[:, [1]]

    full_propensity_test = full_propensity_test.flatten()
    nominal_propensity_test = nominal_propensity_test.flatten()
    odds_ratio_list = ((full_propensity_test / (1 - full_propensity_test))) / (
                nominal_propensity_test / (1 - nominal_propensity_test))

    highest = odds_ratio_list.max()
    lowest = 1 / odds_ratio_list.min()
    true_gamma = max(highest, lowest)
    print(f"true_log_gamma: {true_gamma}")

    return np.log(true_gamma)

# def get_true_propensities(train_ds, test_ds):
#     clf = LogisticRegression(C=1, penalty='elasticnet', solver='saga', l1_ratio=0.7, max_iter=10000)
#     x_full = np.concatenate((train_ds.x, train_ds.u), axis=1)
#     x_full_test = np.concatenate((test_ds.x, test_ds.u), axis=1)
#     clf.fit(x_full, train_ds.t)
#     full_propensity_test = clf.predict_proba(x_full_test)[:, [1]]
#     return full_propensity_test

if __name__ == '__main__':

    project_path = Path(os.getcwd())
    CR_models = []
    method_params_list = []

    # The feature used for creating the synthetic expert
    exp_feature = "work.dur"

    # Path of the directory for saving the results
    ihdp_ver = ""
    ihdp_dir = "ihdp" + ihdp_ver

    trials = 2 #1111

    # Dataset and scores setup
    train_dss = []
    val_dss = []
    test_dss = []
    blearner_results = []
    lce_policies_results = []
    lce_policies_results_pess = []
    pess_policies_results = []
    pt_policies_results = []
    gammas = []

    # calls_list = []

    count_trials = 0
    for trial in range(trials):
        print("-----------", trial, "-----------")
        # Dataset
        ihdp_train_ds = IHDP(root=None, split="train", mode='mu', seed=trial, hidden_confounding=True)
        ihdp_train_ds = update_expert(ihdp_train_ds, feature=exp_feature)
        ihdp_val_ds = IHDP(root=None, split="valid", mode='mu', seed=trial, hidden_confounding=True)
        ihdp_val_ds = update_expert(ihdp_val_ds, feature=exp_feature)
        ihdp_test_ds = IHDP(root=None, split="test", mode='mu', seed=trial, hidden_confounding=True)
        ihdp_test_ds = update_expert(ds=ihdp_test_ds, feature=exp_feature)

        expert_t_train, expert_y_train = get_expert_by_feature(ds=ihdp_train_ds, feature=exp_feature)
        expert_t_test, expert_y_test = get_expert_by_feature(ds=ihdp_test_ds, feature=exp_feature)

        train_dss.append(ihdp_train_ds)
        val_dss.append(ihdp_val_ds)
        test_dss.append(ihdp_test_ds)

        results = compute_all_intervals_BLearner(dir_path=project_path / ihdp_dir,
                                                 trial=trial,
                                                 ds_train=ihdp_train_ds,
                                                 ds_valid=ihdp_val_ds,
                                                 ds_test=ihdp_test_ds)
        blearner_results.append(results)

    # Calculating the B-Learner CAPO bounds in parallel
    #     calls_list.append(delayed(compute_all_intervals_BLearner)(dir_path=project_path / ihdp_dir,
    #                                              trial=trial,
    #                                              ds_train=ihdp_train_ds,
    #                                              ds_valid=ihdp_val_ds,
    #                                              ds_test=ihdp_test_ds))
    #
    # results = Parallel(n_jobs=-2)(calls_list)

    # Policy Value lists of all trials
    pv_oracle_list_all_trials = []
    pv_curr_list_all_trials = []
    pv_lce_logistic_list_all_trials = []
    pv_lce_logistic_list_all_trials_pess = []
    pv_pess_policies_list_all_trials = []
    pv_cr_logit_list_all_trials = []
    pv_cate_interval_list_all_trials = []
    pv_random_defer_list_all_trials = []
    pv_confHAI_policies_list_all_trials = []

    deferral_rates_lce_logistic_list_all_trials = []
    deferral_rates_cate_interval_all_trials = []
    deferral_rates_pess_policies_all_trials = []
    deferral_rate_conhai_list_all_trials = []

    var_CATE = []
    exclude_count = 0

    deferral_rates_random_defer_list = np.arange(0.1, 1.1, 0.1).tolist()

    for trial in range(trials):

        print("-----------------------------", trial, "-----------------------------")

        pv_oracle_list = []
        pv_curr_list = []
        pv_lce_logistic_list = []
        pv_lce_logistic_list_pess = []
        pv_cr_logit_list = []
        pv_cate_interval_list = []
        pv_pess_policies_list = []
        pv_confHAI_policies_list = []

        deferral_rates_cate_interval = []
        deferral_rates_lce_logisitc_list = []
        deferral_rates_pess_policies_list = []
        deferral_rate_conhai_list = []

        # Dataset
        ihdp_train_ds = train_dss[trial]
        ihdp_val_ds = val_dss[trial]
        ihdp_test_ds = test_dss[trial]
        learn_policies_flag = True
        cr_learn_policies = True
        learn_policies_lg_lce_flag = True

        # LCE policies dictionary
        lce_policies = {}
        lce_logistic_policies = {}
        dir_path = project_path / ihdp_dir
        output_dir = dir_path / f"trial-{trial:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        path_lce_logistic = output_dir /"lce_policies"
        path_lce_logistic.mkdir(parents=True, exist_ok=True)
        # path_confhai = output_dir/ "confhai_policies"
        # path_confhai.mkdir(parents=True, exist_ok=True)

        # Paths for saving the policies

        file_path_lce_logistic = output_dir / "lce_policies.json"
        if file_path_lce_logistic.exists():
            with file_path_lce_logistic.open(mode="r") as fp:
                lce_logistic_policies = load_policies(file_path=file_path_lce_logistic)
                learn_policies_flag = False

        file_path_cr = output_dir / "pvs_crlogits"
        if file_path_cr.exists():
            cr_learn_policies = False
            pv_cr_logit_list = pickle.load(open(file_path_cr, "rb"))
            pv_cr_logit_list = pv_cr_logit_list[:len(GAMS)]

        tau_true = torch.tensor(ihdp_test_ds.mu1 - ihdp_test_ds.mu0)
        tau_true_train = torch.tensor(ihdp_train_ds.mu1 - ihdp_train_ds.mu0)
        Y_0 = ihdp_test_ds.y0
        Y_1 = ihdp_test_ds.y1

        # Oracle policy
        oracle_pi = (tau_true > 0) * 1
        oracle_pi_train = (tau_true_train > 0) * 1
        policy_value_oracle = policy_value(oracle_pi, Y_1, Y_0).item()
        pv_oracle_list = [policy_value_oracle] * len(GAMS)

        # Current Policy
        current_policy_train, y_current_expert_train = get_expert_by_feature(ds=ihdp_train_ds, feature=exp_feature)
        current_policy_test, y_current_expert_test = get_expert_by_feature(ds=ihdp_test_ds, feature=exp_feature)
        policy_value_curr_pi = policy_value(current_policy_test, Y_1, Y_0)
        pv_curr_list = [policy_value_curr_pi] * len(GAMS)

        var_CATE_trial = np.sqrt(np.var(ihdp_test_ds.mu1 - ihdp_test_ds.mu0))
        var_CATE.append(var_CATE_trial)
        if var_CATE_trial > 15:
            exclude_count += 1
            continue

        # CR Logit Policy
        if cr_learn_policies:
            # Augmented Covariates
            x_train_ = np.hstack([ihdp_train_ds.x, np.ones([ihdp_train_ds.x.shape[0], 1])])
            x_test_ = np.hstack([ihdp_test_ds.x, np.ones([ihdp_test_ds.x.shape[0], 1])])

            q_0_train_ = ihdp_q0_baseline_p(x_train=x_train_, t_train=ihdp_train_ds.t, x=x_train_).squeeze(1)
            q_0_test_ = ihdp_q0_baseline_p(x_train=x_train_, t_train=ihdp_train_ds.t, x=x_test_).squeeze(1)
            q_0_train = ihdp_q0_baseline_p(x_train=ihdp_train_ds.x, t_train=ihdp_train_ds.t, x=ihdp_train_ds.x).squeeze(
                1)
            q_0_test = ihdp_q0_baseline_p(x_train=ihdp_train_ds.x, t_train=ihdp_train_ds.t, x=ihdp_test_ds.x).squeeze(1)
            baseline_policy = lambda x: ihdp_q0_baseline_p(x_train=x_train_, t_train=ihdp_train_ds.t, x=x)
            def_policy = lambda p: default_policy(q_0=q_0_train_, p=p)
            real_risk_ihdp = lambda p1: real_risk_prob_ihdp(Y1=(-1) * ihdp_test_ds.y1, Y0=(-1) * ihdp_test_ds.y0,
                                                            prob_1=p1)

            # Theta current expert - default policy
            x_all = np.concatenate([ihdp_train_ds.x, ihdp_val_ds.x, ihdp_test_ds.x])
            logit_model = sm.GLM(q_0_test, x_test_, family=sm.families.Binomial())
            # logit_model = sm.GLM(q_0_test, ihdp_test_ds.x, family=sm.families.Binomial())
            logit_result = logit_model.fit()
            coefficients = logit_result.params
            th_expert = coefficients

            opt_config_robust = {'N_RST': 15, 'GRAD_': get_implicit_grad_centered, 'WGHTS_': opt_wrapper,
                                 'GRAD_CTR': get_implicit_grad_centered, 'POL_PROB_1': logistic_pol_asgn,
                                 'POL_GRAD': qk_dpi_dtheta, 'DEFAULT_POL': th_expert,
                                 'BASELINE_POL': def_policy, 'P_1': def_policy, 'averaging': True,
                                 'give_initial': True,
                                 'sharp': True}

            robust_opt_params = {'optimizer': opt_w_restarts, 'pol_opt': 'ogd',
                                 'unc_set_type': 'interval', 'opt_params': opt_config_robust,
                                 'BASELINE_POL': th_expert, 'type': 'logistic-interval'}

            method_params = [robust_opt_params]
            method_params_list.append(method_params)

            test_data = {'x_test': x_test_, 't_test': ihdp_test_ds.t, 'y_test': (-1) * ihdp_test_ds.y,
                         'u_test': (ihdp_test_ds.u).squeeze(1)}
            eval_conf = {'eval': True, 'eval_type': 'ihdp', 'eval_data': test_data, 'oracle_risk': real_risk_ihdp}
            ConfRobPols = [ConfoundingRobustPolicy(baseline_pol=baseline_policy, save=True, verbose=True) for
                           method in
                           method_params]
            for ind, method_param in enumerate(method_params):
                ConfRobPols[ind].fit(x_train_, ihdp_train_ds.t, (-1) * ihdp_train_ds.y, q_0_train_, GAMS, method_param,
                                     eval_conf=eval_conf)
            # CR_models.append(ConfRobPols)
            pv_cr_logit_list = -1 * (ConfRobPols[0].PVS)
            pickle.dump(pv_cr_logit_list, open(file_path_cr, "wb"))
        tau_mean_test_list = []
        pvs_random_defer_per_gamma = []
        for k, v in GAMMAS.items():

            log_gamma = k
            gamma = v
            print(k)
            pv_random_defer_list = []

            # Estimating CATE and CAPO bounds using B-Learner
            tau_hat = blearner_results[trial][log_gamma]  # send results[trial] to the update sens
            tau_mean_train, tau_bottom_train, tau_top_train, \
            tau_mean_test, tau_bottom_test, tau_top_test, \
            tau_mean_val, tau_bottom_val, tau_top_val, \
            Y_0_bottom_train, Y_0_top_train, Y_1_bottom_train, Y_1_top_train, \
            Y_0_bottom_test, Y_0_top_test, Y_1_bottom_test, Y_1_top_test, \
            Y_0_bottom_val, Y_0_top_val, Y_1_bottom_val, Y_1_top_val = extract_results(tau_hat)

            # Random Defer Policy
            n_test = len(ihdp_test_ds.x)
            current_expet_test = torch.Tensor(current_policy_test).squeeze(0)

            for def_rate in deferral_rates_random_defer_list:
                pi_random_defer = (tau_mean_test > 0) * 1
                pi_random_defer = pi_random_defer.squeeze(0)
                sub_n = math.floor(def_rate * n_test)
                deferral_indicies = random.sample(range(n_test), k=sub_n)
                for j in range(len(pi_random_defer)):
                    if j in deferral_indicies:
                        pi_random_defer[j] = current_expet_test[j]
                policy_value_random_defer = policy_value(pi=pi_random_defer, y1=Y_1, y0=Y_0).item()
                pv_random_defer_list.append(policy_value_random_defer)
            pvs_random_defer_per_gamma.append(pv_random_defer_list)

            # LCE Policy Class
            propensity_model, \
            quantile_model_lower, quantile_model_upper, \
            cvar_model_lower, cvar_model_upper, \
            mu_model, \
            cate_bounds_model = get_nuisances_models_RF(gamma=gamma)

            # Logistic Policy model
            lce_logistic_model = nn.Sequential(
                nn.Linear(24, 3),
                nn.Sigmoid()
            )

            lce_policy_path = path_lce_logistic/ f"lce_{log_gamma}.pkl"
            if learn_policies_flag:
                lce_logistic_policy_model = LCE_Policy(tau_hat=tau_hat,
                                                       propensity_model=propensity_model,
                                                       quantile_minus_model=quantile_model_lower,
                                                       quantile_plus_model=quantile_model_upper,
                                                       cvar_minus_model=cvar_model_lower,
                                                       cvar_plus_model=cvar_model_upper,
                                                       mu_model=mu_model,
                                                       policy_model=lce_logistic_model,
                                                       cate_bounds_model=cate_bounds_model,
                                                       use_rho=True,
                                                       gamma=gamma,
                                                       higher_better=True)

                lce_logistic_policy_model.fit(ds_train=ihdp_train_ds, ds_valid=ihdp_val_ds, devices=[0,1])
                lce_logistic_pi = lce_logistic_policy_model.predict(ds_test=ihdp_test_ds)
            else:
                lce_logistic_pi = lce_logistic_policies[k]

            lce_logistic_pi_with_exp, defer_count_logistic, deferral_rate_logistic = update_expert_preds(
                preds=lce_logistic_pi,
                expert_labels=torch.Tensor(
                    current_policy_test).type(
                    torch.LongTensor))

            lce_logistic_pv = policy_value(pi=lce_logistic_pi_with_exp, y1=Y_1, y0=Y_0)
            pv_lce_logistic_list.append(lce_logistic_pv)
            print(f"LCE Policy Value: {lce_logistic_pv}")
            deferral_rates_lce_logisitc_list.append(deferral_rate_logistic)
            # Add policy to policies dictionary
            if learn_policies_lg_lce_flag:
                lce_logistic_policies.update({k: torch.Tensor(lce_logistic_pi)})

            # CATE Policy
            current_expet_test = torch.Tensor(current_policy_test).squeeze(0)
            pi_cate_interval = ((tau_top_test > 0) * (tau_bottom_test > 0)) * 1
            defer_indxs = np.where(((tau_top_test >= 0) * (tau_bottom_test <= 0)) * 1)[1]

            pi_cate_interval = pi_cate_interval.squeeze(0)
            for j in range(len(pi_cate_interval)):
                if j in defer_indxs:
                    pi_cate_interval[j] = current_expet_test[j]
            policy_value_cate_interval = policy_value(pi=pi_cate_interval, y1=Y_1, y0=Y_0).item()
            pv_cate_interval_list.append(policy_value_cate_interval)
            deferral_rates_cate_interval.append(len(defer_indxs) / len(pi_cate_interval))

            # Pessimistc Policy
            pi_pess_policies = np.zeros(len(current_expet_test))
            for j in range(len(pi_pess_policies)):
                if (Y_1_bottom_test.squeeze(0)[j] - Y_0_top_test.squeeze(0)[j]) > 0:
                    pi_pess_policies[j] = 1
                elif (Y_1_top_test.squeeze(0)[j] - Y_0_bottom_test.squeeze(0)[j]) < 0:
                    pi_pess_policies[j] = 0
                elif Y_1_bottom_test.squeeze(0)[j] > Y_0_bottom_test.squeeze(0)[j]:
                    pi_pess_policies[j] = 1
                else:
                    pi_pess_policies[j] = 0

            policy_value_pess_policies = policy_value(pi=pi_pess_policies, y1=Y_1, y0=Y_0).item()
            pv_pess_policies_list.append(policy_value_pess_policies)


        if learn_policies_flag:
            save_policies(policies=lce_logistic_policies, file_path=file_path_lce_logistic)

        # Policy Value lists for all trials
        pv_oracle_list_all_trials.append(pv_oracle_list)
        pv_curr_list_all_trials.append(pv_curr_list)
        pv_lce_logistic_list_all_trials.append(pv_lce_logistic_list)
        pv_cr_logit_list_all_trials.append(pv_cr_logit_list)
        pv_cate_interval_list_all_trials.append(pv_cate_interval_list)
        pv_pess_policies_list_all_trials.append(pv_pess_policies_list)
        pv_confHAI_policies_list_all_trials.append(pv_confHAI_policies_list)

        pv_random_defer_list_all_trials.append(np.mean(pvs_random_defer_per_gamma, axis=0))

        # Deferral rates lists for all trials
        deferral_rates_lce_logistic_list_all_trials.append(deferral_rates_lce_logisitc_list)
        deferral_rates_cate_interval_all_trials.append(deferral_rates_cate_interval)
        deferral_rate_conhai_list_all_trials.append(deferral_rate_conhai_list)

    deferral_rates_lce_logistic_means, deferral_rates_lce_logistic_sem = calc_means_sems(
        deferral_rates_lce_logistic_list_all_trials)

    deferral_rates_confhai_means, deferral_rates_confhai_sem = calc_means_sems(deferral_rate_conhai_list_all_trials)
    deferral_rates = [i / (len(pv_oracle_list_all_trials[0]) - 1) for i in
                      range(len(pv_oracle_list_all_trials[0]))] if len(pv_oracle_list_all_trials[0]) > 1 else [1.0]

    deferral_rates_cate_interval_mean, _ = calc_means_sems(deferral_rates_cate_interval_all_trials)

    # Calculating true log gamma
    true_log_gamma = get_ihdp_true_gamma(ihdp_train_ds=train_dss[0], ihdp_test_ds=test_dss[0])

    # Plotting the results
    colors = ['b', 'g', 'r', 'm', 'b', 'purple', 'brown', 'c', ]
    pltlabels = ['CRLogit', 'CRLogit L1 0.5', 'CRLogit L1 0.25']
    markers = ['>', '+', '.', ',', 'o', 'v', 'x', 's', 'D', '|']
    fig = plt.figure(figsize=(8, 3))

    # Policy value - log gamma plot
    ax = ggplot_log_style_deferral(figsize=(682 / 72, 512 / 72), log_y=False)



    means, sds = calc_means_sems(pv_curr_list_all_trials)
    plt.plot(GAMS, means, label="Expert Policy", color=colors[1], marker=markers[1])
    plt.fill_between(GAMS, means - sds, means + sds, color=colors[1], alpha=0.2)

    means, sds = calc_means_sems(pv_cr_logit_list_all_trials)
    plt.plot(GAMS, means, label="CRLogit Policy", color=colors[3], marker=markers[2])
    plt.fill_between(GAMS, means - sds, means + sds, color=colors[3], alpha=0.2)

    # means, sds = calc_means_sems(pv_confHAI_policies_list_all_trials)
    # plt.plot(GAMS, means, label="CONfHAI Policy", color="crimson", marker=markers[6])
    # plt.fill_between(GAMS, means - sds, means + sds, color="crimson", alpha=0.2)

    # means, sds = calc_means_sems(pv_lce_logistic_list_all_trials)
    # plt.plot(GAMS, means, label="CARED Policy", color="C1", marker=markers[4])
    # plt.fill_between(GAMS, means - sds, means + sds, color="C1", alpha=0.2)

    means, sds = calc_means_sems(pv_pess_policies_list_all_trials)
    plt.plot(GAMS, means, label="Pessimistic Policy", color="navy", marker=markers[5])
    plt.fill_between(GAMS, means - sds, means + sds, color="navy", alpha=0.2)

    means, sds = calc_means_sems(pv_cate_interval_list_all_trials)
    plt.plot(GAMS, means, label="B-Learner Policy", color=colors[5], marker=markers[3])
    plt.fill_between(GAMS, means - sds, means + sds, color=colors[5], alpha=0.2)

    means, sds = calc_means_sems(pv_oracle_list_all_trials)
    plt.plot(GAMS, means, label="Oracle Policy", color=colors[0], marker=markers[0])
    plt.fill_between(GAMS, means - sds, means + sds, color=colors[0], alpha=0.2)

    plt.axvline(x=true_log_gamma, color='black', label=r'True $\log(\Lambda)$')

    plt.xscale('log')
    ax.set_ylabel("Policy Value")
    # plt.legend(loc=2)
    plt.legend(fontsize=12, loc=(1.02, 0.42))
    plt.xlabel(r'$\log(\Lambda)$ uncertainty parameter', fontsize=15)
    # plt.legend()
    # plt.savefig("ihdp_policy_value_log_gamma_plot_updated.pdf")
    plt.show()

    # Deferral Rates plot
    ax = ggplot_log_style_deferral(figsize=(682 / 72, 512 / 72), log_y=False)

    # Expert's Policy
    means, sds = calc_means_sems(pv_curr_list_all_trials)
    plt.plot(deferral_rates, means, label="Expert Policy", color=colors[1], marker=markers[1])
    plt.fill_between(deferral_rates, means - sds, means + sds, color=colors[1], alpha=0.2)

    # # CONFHAI Policy
    # means, sds = calc_means_sems(pv_confHAI_policies_list_all_trials)
    # plt.scatter(deferral_rates_confhai_means, means, label="CONFHAI Policy", color="crimson", marker='D')
    # plt.errorbar(deferral_rates_confhai_means, means,
    #              yerr=sds,
    #              fmt='o', color="crimson")

    # # LCE Policy - Logistic
    # means, sds = calc_means_sems(pv_lce_logistic_list_all_trials)
    # plt.scatter(deferral_rates_lce_logistic_means, means, label="CARED Policy", color="C1", marker=markers[4])
    # plt.errorbar(deferral_rates_lce_logistic_means, means,
    #              yerr=sds,
    #              fmt='o', color="C1")


    # B-Learner Policy
    means, sds = calc_means_sems(pv_cate_interval_list_all_trials)
    plt.scatter(deferral_rates_cate_interval_mean, means, label="B-Learner Policy", color=colors[5], marker=markers[3])
    plt.errorbar(deferral_rates_cate_interval_mean, means,
                 yerr=sds,
                 fmt='o', color=colors[5])
    # Random Defer Policy
    means, sds = calc_means_sems(pv_random_defer_list_all_trials)
    plt.scatter(deferral_rates_random_defer_list, means, label="Random Deferral Policy", color="maroon",
                marker=markers[5])
    plt.errorbar(deferral_rates_random_defer_list, means,
                 yerr=sds,
                 fmt='o', color="maroon")
    # Oracle Policy
    means, sds = calc_means_sems(pv_oracle_list_all_trials)
    plt.plot(deferral_rates, means, label="Oracle Policy", color=colors[0], marker=markers[0])
    plt.fill_between(deferral_rates, means - sds, means + sds, color=colors[0], alpha=0.2)

    ax.set_ylabel("Policy Value")
    ax.set_xlabel("Deferral Rate")
    # plt.legend(loc=2)
    plt.legend(fontsize=12, loc=(1.02, 0.42))
    # plt.legend()
    # plt.savefig("ihdp_policy_value_deferral_plot_updated_.pdf")
    plt.show()
