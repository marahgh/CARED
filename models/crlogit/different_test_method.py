# def gen_data_run_for_gamma_for_joblib(dgp_params, GAMS, real_risk_prob, method_params, ind_rep, gen_data=True,
#                                       save=False, save_params=[], already_gen_data=[]):
#     print ind_rep
#     if gen_data:
#         [x_full, u, T_, Y_, true_Q_, q0] = generate_log_data(**dgp_params)
#     else:
#         [x_full, u, T_, Y_, true_Q_, q0] = already_gen_data
#     np.random.seed(ind_rep)
#     random.seed(ind_rep)
#     train_ind, test_ind = model_selection.train_test_split(range(len(Y_)), test_size=0.9)
#     test_data = {'x_test': x_full[test_ind, :], 't_test': T_[test_ind], 'y_test': Y_[test_ind], 'u_test': u[test_ind]}
#     eval_conf = {'eval': True, 'eval_type': 'true_dgp', 'eval_data': test_data, 'oracle_risk': real_risk_prob}
#     ConfRobPols = [ConfoundingRobustPolicy(baseline_pol=ctrl_p_1, save_params=save_params, save = True, verbose=True) for method in
#                    method_params]
#     for ind, method_param in enumerate(method_params):
#         ConfRobPols[ind].fit(x_full[train_ind, :], T_[train_ind], Y_[train_ind], q0[train_ind], GAMS, method_param,
#                              eval_conf=eval_conf)
#         del ConfRobPols[ind].x
#         del ConfRobPols[ind].t
#         del ConfRobPols[ind].y
#         del ConfRobPols[ind].eval_data
#
#     return ConfRobPols
#
# res = Parallel(n_jobs=-1, verbose=40)(
#     delayed(gen_data_run_for_gamma_for_joblib)(dgp_params, GAMS, real_risk_prob, method_params, i, gen_data=True,
#                                                save=True, save_params=save_params) for i in range(N_REPS))
