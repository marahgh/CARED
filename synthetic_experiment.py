from pathlib import Path
import json
import os
from joblib import Parallel, delayed
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from sklearn import model_selection

from compute_blearner_bounds_ihdp import policy_value
from datasets.confhai_data import *
from compute_blearner_bounds_synthetic import compute_all_intervals_BLearner
from models.lce_surrogate.lce_model_pl import LCEModel
from models.lce_policy.lce_policy import LCE_Policy
from datasets.synthetic_log_confhai import SYN_LOG

from matplotlib import rcParams
import seaborn as sns
from datasets.confhai_data import seed_everything

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
import matplotlib.pyplot as plt
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



def policy_regret(pi, y0, y1):
    pv =policy_value(pi, y0=y0, y1=y1)
    baseline_pv = y0.mean()
    return pv  - baseline_pv

GAMS = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
c = 0
nrep = 10
wgamma = np.array([2.5, 2.5, 2.5]).ravel()
hidd = 2

d = 5  # dimension of x
n = 2000
ntest = 10000
# parameters
rho = np.asarray([1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0, 1 / np.sqrt(3)])  # normalize to unit 0.5
rho = rho / (np.dot(rho, rho) * 2)

beta_cons = 2.5
beta_x = np.asarray([0, .5, -0.5, 0, 0, 0])  # beta_0
beta_x_T = np.asarray([-1.5, 1, -1.5, 1., 0.5, 0])  # beta_treat
beta_T = np.asarray([0, .75, -.5, 0, -1, 0, 0])  # beta
beta_T_conf = np.asarray([0, .75, -.5, 0, -1, 0])
# beta_T = np.asarray([-1, 0,0, -.5, 1,0.5,1.5])
# beta_T_conf = np.asarray([-1, 0,0, -.5, 1,0.5 ])
mu_x = np.asarray([-1, .5, -1, 0, -1]);

alpha = -2
w = 1.5
# true specified human model
Gamma = wgamma

dgp_params = {
    "mu_x": mu_x, "n": n, "beta_cons": beta_cons, "beta_x": beta_x, "beta_x_T": beta_x_T,
    "beta_T_conf": beta_T_conf, "Gamma": Gamma, "alpha": alpha, "w": w
}

dgp_params_test = {
    "mu_x": mu_x, "n": ntest, "beta_cons": beta_cons, "beta_x": beta_x, "beta_x_T": beta_x_T,
    "beta_T_conf": beta_T_conf, "Gamma": Gamma, "alpha": alpha, "w": w
}
# Plots Setup
ax = ggplot_log_style_deferral(figsize=(682 / 72, 512 / 72), log_y=False)
colors = ['b', 'g', 'r', 'm', 'b', 'purple', 'brown', 'c', ]
markers = ['>', '+', '.', ',', 'o', 'v', 'x', 's', 'D', '|']


def save_policies(policies, file_path):
    tmp_policies = policies
    for k, v in tmp_policies.items():
        tmp_policies[k] = (v.numpy()).tolist()
    with file_path.open(mode="w") as fp:
        json.dump(tmp_policies, fp)


def load_policies(file_path):
    with file_path.open(mode="r") as fp:
        policies = json.load(fp)
    for k, v in policies.items():
        policies[k] = torch.Tensor(v)
    return policies


def get_deferral_costs(y_expert, a_expert, y0_hat, y1_hat):
    deferral_costs = []
    for j in range(len(a_expert)):
        if a_expert[j] == 0:
            deferral_costs.append(y1_hat[j] - y_expert[j])
        else:
            deferral_costs.append(y0_hat[j] - y_expert[j])

    return torch.Tensor(deferral_costs).unsqueeze(0)


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


def extract_results(res_tau_hat):
    res_tau_mean_train = res_tau_hat["tau_mean_train"]
    res_tau_bottom_train = res_tau_hat["tau_bottom_train"]
    res_tau_top_train = res_tau_hat["tau_top_train"]
    res_tau_mean_test = res_tau_hat["tau_mean_test"]
    res_tau_bottom_test = res_tau_hat["tau_bottom_test"]
    res_tau_top_test = res_tau_hat["tau_top_test"]
    res_Y_0_bottom_train = res_tau_hat["Y_0_bottom_train"]
    res_Y_0_top_train = res_tau_hat["Y_0_top_train"]
    res_Y_1_bottom_train = res_tau_hat["Y_1_bottom_train"]
    res_Y_1_top_train = res_tau_hat["Y_1_top_train"]
    res_Y_0_bottom_test = res_tau_hat["Y_0_bottom_test"]
    res_Y_0_top_test = res_tau_hat["Y_0_top_test"]
    res_Y_1_bottom_test = res_tau_hat["Y_1_bottom_test"]
    res_Y_1_top_test = res_tau_hat["Y_1_top_test"]

    return res_tau_mean_train, res_tau_bottom_train, res_tau_top_train, \
           res_tau_mean_test, res_tau_bottom_test, res_tau_top_test, \
           res_Y_0_bottom_train, res_Y_0_top_train, res_Y_1_bottom_train, res_Y_1_top_train, \
           res_Y_0_bottom_test, res_Y_0_top_test, res_Y_1_bottom_test, res_Y_1_top_test


def get_confhai_result(results_dir):
    human_per_log_gamma = []
    ao_per_log_gamma = []
    confAo_per_log_gamma = []
    hAi_per_log_gamma = []
    confHAi_per_log_gamma = []
    confHAiPerson_per_log_gamma = []

    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith(".csv"):
            print(filename)
            file_path = os.path.join(results_dir, filename)

            # Read the CSV file
            data = pd.read_csv(file_path)
            human_per_log_gamma.append(np.asarray(data["Human"]))

            ao_per_log_gamma.append(np.asarray(data["AO"]))
            confAo_per_log_gamma.append(np.asarray(data["ConfAO"]))
            hAi_per_log_gamma.append(np.asarray(data["HAI"]))
            confHAi_per_log_gamma.append(np.asarray(data["ConfHAI"]))
            # confHAiPerson_per_log_gamma.append(np.asarray(data["ConfHAIPerson"]))

    labels = ["Human's Policy", "AO", "CRLogit Policy", "HAI", "ConfHAI Policy"]
    results = [human_per_log_gamma, ao_per_log_gamma, confAo_per_log_gamma, hAi_per_log_gamma, confHAi_per_log_gamma]
    return results, labels


if __name__ == '__main__':
    project_path = Path(os.getcwd())
    dir_path = project_path / "syn_log_confhai_exp"

    num_teatments = 2

    confhai_results_dir = project_path / "confhai_results"

    confhai_results, confhai_labels = get_confhai_result(results_dir=confhai_results_dir)
    # confhai_colors = ['blue', 'orange', 'green', 'red', 'purple']
    confhai_colors = ['g', 'orange', 'm', 'red', 'crimson']
    confhai_markers = ['o', '^', 'v', '*', 'x']
    for i in range(len(confhai_labels)):
        if confhai_labels[i] == "AO" or confhai_labels[i] == "HAI":
            continue
        result = confhai_results[i]
        means = np.mean(np.asarray(result), axis=1)
        sds = np.std(np.asarray(result), axis=1) / np.sqrt(nrep)
        plt.plot(GAMS, means, label=confhai_labels[i], color=confhai_colors[i], marker=confhai_markers[i])
        plt.fill_between(GAMS, means -sds, means + sds, color=confhai_colors[i], alpha=0.1)

    pv_conf_cate_list_all_trials = []
    pv_all_control_list_all_trials = []
    pv_lce_list_all_trials = []
    pv_lce_list_all_trials_pess = []
    pv_cate_interval_list_all_trials = []
    pv_oracle_list_all_trails = []
    pv_random_defer_list_all_trials = []
    pv_pess_policies_list_all_trials = []

    calls_list = []
    deferral_rates_random_defer_list = np.arange(0.1, 1.1, 0.1).tolist()

    for trial in range(nrep):
        print("---------------------------", trial, "---------------------------")
        seed_everything(trial)
        learn_policies_lg_lce_flag = True
        learn_policies_lg_lce_flag_pess = True
        lce_logistic_policies = {}
        lce_logistic_policies_pess = {}
        output_dir = dir_path / f"trial-{trial:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Paths for saving the policies
        file_path_lce_logistic = output_dir / "lce_policies.json"
        file_path_lce_logistic_pess = output_dir / "lce_policies_no_deferral.json"

        if file_path_lce_logistic.exists():
            with file_path_lce_logistic.open(mode="r") as fp:
                lce_logistic_policies = load_policies(file_path=file_path_lce_logistic)
                learn_policies_lg_lce_flag = False

        if file_path_lce_logistic_pess.exists():
            with file_path_lce_logistic_pess.open(mode="r") as fp:
                lce_logistic_policies_pess = load_policies(file_path=file_path_lce_logistic_pess)
                learn_policies_lg_lce_flag_pess = False
        # Datasets
        ds_train = SYN_LOG(dgp_params=dgp_params, split="train", seed=trial)
        ds_test = SYN_LOG(dgp_params=dgp_params_test, split="test", seed=trial)
        Y0_test = ds_test.y0
        Y1_test = ds_test.y1
        expert_test_t = np.zeros(len(ds_test.t))
        expert_test_y = ds_test.y0

        # CAPO bounds estimation
        tau_hat_all_gams = compute_all_intervals_BLearner(dir_path=dir_path,
                                                          trial=trial,
                                                          ds_train=ds_train,
                                                          ds_test=ds_test)

        # Estimate CAPO Bounds in Parallel

    #     calls_list.append(delayed(compute_all_intervals_BLearner(dir_path=dir_path,
    #                                                       trial=trial,
    #                                                       ds_train=ds_train,
    #                                                       ds_test=ds_test)))
    #
    # results = Parallel(n_jobs=-2)(calls_list)
        # Policy Value lists per trial
        pv_conf_cate_list = []
        pv_all_control_list = []
        pv_cate_interval_list = []
        pv_lce_list = []
        pv_lce_list_pess = []
        pv_pess_policies_list = []
        pv_oracle_list = []

        pvs_random_defer_per_gamma = []
        for ind_g, gamma in enumerate(GAMS):
            print(gamma)
            pv_random_defer_list = []

            # pv_all_control = np.mean(real_risk_prob(np.zeros(len(ds_test.t)), ds_test.x, ds_test.u))
            # pv_all_control_list.append(pv_all_control)

            pi_oracle = (Y1_test < Y0_test) * 1
            pv_oracle = policy_regret(pi=pi_oracle, y0=Y0_test, y1=Y1_test)
            pv_oracle_list.append(pv_oracle)

            tau_hat = tau_hat_all_gams[str(gamma)]

            tau_mean_train, tau_bottom_train, tau_top_train, \
            tau_mean_test, tau_bottom_test, tau_top_test, \
            Y_0_bottom_train, Y_0_top_train, Y_1_bottom_train, Y_1_top_train, \
            Y_0_bottom_test, Y_0_top_test, Y_1_bottom_test, Y_1_top_test = extract_results(tau_hat)

            # Random Defer Policy
            n_test = len(ds_test.x)
            current_expet_test = torch.Tensor(expert_test_t).squeeze(0)

            for def_rate in deferral_rates_random_defer_list:
                pi_random_defer = (tau_mean_test > 0) * 1
                pi_random_defer = pi_random_defer.squeeze(0)
                sub_n = math.floor(def_rate * n_test)
                deferral_indicies = random.sample(range(n_test), k=sub_n)
                for j in range(len(pi_random_defer)):
                    if j in deferral_indicies:
                        pi_random_defer[j] = current_expet_test[j]
                # policy_value_random_defer = np.mean(real_risk_prob(pi_random_defer, ds_test.x, ds_test.u))
                # pv_random_defer_list.append(policy_value_random_defer)
            # pvs_random_defer_per_gamma.append(pv_random_defer_list)

            pi_conf_cate = (tau_mean_test < 0) * 1
            pv_conf_cate = policy_regret(pi=pi_conf_cate, y0=Y0_test, y1=Y1_test)
            pv_conf_cate_list.append(pv_conf_cate)


            # CATE Policy
            pi_cate_interval = ((tau_top_test < 0) * (tau_bottom_test < 0)) * 1
            defer_indxs = np.where(((tau_top_test >= 0) * (tau_bottom_test <= 0)) * 1)[1]

            pi_cate_interval = pi_cate_interval.squeeze(0)
            for j in range(len(pi_cate_interval)):
                if j in defer_indxs:
                    pi_cate_interval[j] = 0
            policy_value_cate_interval = policy_regret(pi=pi_cate_interval, y0=Y0_test, y1=Y1_test)
            pv_cate_interval_list.append(policy_value_cate_interval)

            treatment_costs = Y_1_top_train - Y_0_bottom_train
            control_costs = Y_0_top_train - Y_1_bottom_train

            # deferral_costs_cons = -1*get_deferral_costs(y0_hat=Y_0_bottom_train.squeeze(0),
            #                                              y1_hat=Y_1_bottom_train.squeeze(0),
            #                                              y_expert=ds_train.y, a_expert=ds_train.t)
            deferral_costs_cons = -1 * get_deferral_costs(y0_hat=Y_0_bottom_train.squeeze(0),
                                                          y1_hat=Y_1_bottom_train.squeeze(0),
                                                          y_expert=ds_train.y0, a_expert=np.zeros(len(ds_train.t)))

            gamma_costs_matrix_cons = (np.concatenate(((np.concatenate((control_costs, treatment_costs
                                                                        ), axis=0)), deferral_costs_cons),
                                                      axis=0)).transpose()

            gamma_costs_matrix_cons_pess = (np.concatenate((control_costs, treatment_costs
                                                            ), axis=0)).transpose()

            deferral_costs_opt = -1 * get_deferral_costs(y0_hat=Y_0_top_train.squeeze(0),
                                                         y1_hat=Y_1_top_train.squeeze(0),
                                                         y_expert=ds_train.y, a_expert=ds_train.t)

            gamma_costs_matrix_opt = (np.concatenate(((np.concatenate((control_costs, treatment_costs
                                                                       ), axis=0)), deferral_costs_opt),
                                                     axis=0)).transpose()

            model = nn.Sequential(
                nn.Linear(5, 3),
                nn.Sigmoid()
            )
            if learn_policies_lg_lce_flag:
                train_loader = DataLoader(ds_train, batch_size=16, shuffle=True, drop_last=True)
                test_loader = DataLoader(ds_test, batch_size=10, shuffle=True, drop_last=True)
                early_stop_callback = EarlyStopping(monitor="train_loss", patience=3, verbose=False,
                                                    mode="min")
                trainer = Trainer(callbacks=[early_stop_callback],
                                  max_epochs=100, log_every_n_steps=2, accelerator="gpu", devices=[0])
                lce_model = LCEModel(pmodel=model, training_costs=gamma_costs_matrix_cons,
                                     lr=0.001, weight_decay=0)
                trainer.fit(model=lce_model, train_dataloaders=train_loader)

                lce_pi = lce_model.predict(test_loader)
                lce_logistic_policies.update({gamma: torch.Tensor(lce_pi)})
                lce_logistic_policy_model = LCE_Policy(tau_hat=tau_hat,
                                                       propensity_model=None,
                                                       quantile_minus_model=None,
                                                       quantile_plus_model=None,
                                                       cvar_minus_model=None,
                                                       cvar_plus_model=None,
                                                       mu_model=None,
                                                       policy_model=model,
                                                       cate_bounds_model=None,
                                                       use_rho=True,
                                                       gamma=gamma,
                                                       higher_better=False,
                                                       baseline_p0=True)
                lce_logistic_policy_model.fit(ds_train=ds_train, ds_valid=ds_valid)
                lce_pi = lce_logistic_policy_model.predict(ds_test=ds_test)
                lce_logistic_policies.update({gamma: torch.Tensor(lce_pi)})
            else:
                lce_pi = lce_logistic_policies[str(gamma)]
            lce_pi_with_exp, defer_count, deferral_rate = update_expert_preds(preds=lce_pi,
                                                                              expert_labels=torch.Tensor(
                                                                                  np.zeros(len(ds_test.t))).type(
                                                                                  torch.LongTensor))
            pv_lce = policy_regret(pi=lce_pi_with_exp, y0=Y0_test, y1=Y1_test)
            pv_lce_list.append(pv_lce)


            # Pessimistc Policy
            pi_pess_policies = np.zeros(len(current_expet_test))
            for j in range(len(pi_pess_policies)):
                if (Y_1_top_test.squeeze(0)[j] - Y_0_bottom_test.squeeze(0)[j]) < 0:
                    pi_pess_policies[j] = 1
                elif (Y_1_bottom_test.squeeze(0)[j] - Y_0_top_test.squeeze(0)[j]) > 0:
                    pi_pess_policies[j] = 0
                elif Y_1_top_test.squeeze(0)[j] < Y_0_top_test.squeeze(0)[j]:
                    pi_pess_policies[j] = 1
                else:
                    pi_pess_policies[j] = 0
            policy_value_pess_policies = policy_regret(pi=torch.Tensor(pi_pess_policies), y1=Y1_test, y0=Y0_test).item()
            pv_pess_policies_list.append(policy_value_pess_policies)

        if learn_policies_lg_lce_flag:
            save_policies(policies=lce_logistic_policies, file_path=file_path_lce_logistic)
        if learn_policies_lg_lce_flag_pess:
            save_policies(policies=lce_logistic_policies_pess, file_path=file_path_lce_logistic_pess)

        pv_oracle_list_all_trails.append(pv_oracle_list)
        pv_conf_cate_list_all_trials.append(pv_conf_cate_list)
        pv_all_control_list_all_trials.append(pv_all_control_list)
        pv_lce_list_all_trials.append(pv_lce_list)
        pv_lce_list_all_trials_pess.append(pv_lce_list_pess)
        pv_pess_policies_list_all_trials.append(pv_pess_policies_list)
        pv_cate_interval_list_all_trials.append(pv_cate_interval_list)


    confhai_colors = ['b', 'g', 'r', 'm', 'b', 'purple', 'brown', 'c', ]


    plt.plot(GAMS, np.mean(pv_lce_list_all_trials, axis=0), label="CARED Policy", color="C1", marker=markers[3])
    plt.fill_between(GAMS,
                     np.mean(pv_lce_list_all_trials, axis=0) - np.std(pv_lce_list_all_trials, axis=0) / np.sqrt(nrep)
                     ,
                     np.mean(pv_lce_list_all_trials, axis=0) + np.std(pv_lce_list_all_trials, axis=0) / np.sqrt(nrep),
                     color="C1", alpha=0.1)

    plt.plot(GAMS, np.mean(pv_lce_list_all_trials_pess, axis=0), label="CARED Policy - No Deferral", color="navy",
             marker=markers[4])
    plt.fill_between(GAMS,
                     np.mean(pv_lce_list_all_trials_pess, axis=0) - np.std(pv_lce_list_all_trials_pess,
                                                                           axis=0) / np.sqrt(nrep)
                     ,
                     np.mean(pv_lce_list_all_trials_pess, axis=0) + np.std(pv_lce_list_all_trials_pess,
                                                                           axis=0) / np.sqrt(nrep),
                     color="navy", alpha=0.1)

    plt.plot(GAMS, np.mean(pv_pess_policies_list_all_trials, axis=0), label="Pessimistic Policy", color="navy",
             marker=markers[4])
    plt.fill_between(GAMS,
                     np.mean(pv_pess_policies_list_all_trials, axis=0) - np.std(pv_pess_policies_list_all_trials,
                                                                           axis=0) / np.sqrt(nrep)
                     ,
                     np.mean(pv_pess_policies_list_all_trials, axis=0) + np.std(pv_pess_policies_list_all_trials,
                                                                           axis=0) / np.sqrt(nrep),
                     color="navy", alpha=0.1)


    plt.plot(GAMS, np.mean(pv_cate_interval_list_all_trials, axis=0), label="B-Learner Policy", color=colors[5],
             marker=markers[6])
    plt.fill_between(GAMS,
                     np.mean(pv_cate_interval_list_all_trials, axis=0) - np.std(pv_cate_interval_list_all_trials,
                                                                                axis=0) / np.sqrt(nrep)
                     , np.mean(pv_cate_interval_list_all_trials, axis=0) + np.std(pv_cate_interval_list_all_trials,
                                                                                  axis=0) / np.sqrt(nrep),
                     color=colors[5], alpha=0.1)

    plt.plot(GAMS, np.mean(pv_oracle_list_all_trails, axis=0), label="Oracle Policy", color=colors[0], marker=markers[7])
    plt.fill_between(GAMS,
                     np.mean(pv_oracle_list_all_trails, axis=0) - np.std(pv_oracle_list_all_trails,
                                                                         axis=0) / np.sqrt(nrep)
                     , np.mean(pv_oracle_list_all_trails, axis=0) + np.std(pv_oracle_list_all_trails,
                                                                           axis=0) / np.sqrt(nrep),
                     color=colors[0], alpha=0.1)

    plt.axvline(x=2.5, color='black', label=r'True $\log(\Lambda)$')
    plt.ylabel('Policy Regret', fontsize=15)
    plt.xlabel(r'$\log(\Lambda)$ uncertainty parameter', fontsize=15)
    # plt.legend(loc=3)
    plt.legend(fontsize=12, loc=(1.02, 0.42))
    plt.savefig('synthetic_policy_value_log_gamma.pdf', bbox_inches='tight')
    plt.show()
