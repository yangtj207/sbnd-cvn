import matplotlib.pylab as plt
import pickle
import numpy as np
import h5py
import re

path = '/home/nitish/data'
def getData(variant, old=False):
    global path
    data = np.load(path+'/cvn_gaushit/3view/output/multioutput_8x6_fid.np', 'rb', allow_pickle=True)
    if variant == 'cvn_hd' and not old:
        data = np.load(path+'/'+variant+'/3view/output/multioutput_2x6_hd_fid.np', 'rb', allow_pickle=True)
    if old:
        data = np.load(path+'/'+variant+'/3view/output/multioutput_2x6_hd_fid_old.np', 'rb', allow_pickle=True)
    data = dict(enumerate(data.flatten(), 1))
    return data[1]

def getIDs(variant):
    global path
    data = pickle.load(open(path+'/'+variant+'/3view/pickle/labels_8x6_fid.p', 'rb'))
    data_partition = pickle.load(open(path+'/'+variant+'/3view/pickle/partition_8x6_fid.p', 'rb'))
    if variant == 'cvn_hd':
        data = pickle.load(open(path+'/'+variant+'/3view/pickle/labels_2x6_hd_fid.p', 'rb'))
        data_partition = pickle.load(open(path+'/'+variant+'/3view/pickle/partition_2x6_hd_fid.p', 'rb'))

    return data, data_partition

#  cvn_vd_pots = {'numucc':9.7859664, 'nuecc':8.4888697, 'nutaucc':26.672228, 'NC':9.7859664}
#  cvn_hd_pots = {'numucc':16.054934, 'nuecc':16.138479, 'nutaucc':51.384452, 'NC':16.054934}
cvn_vd_pots = [9.7859664, 8.4888697, 26.672228, 9.7859664]
cvn_hd_pots = [6.4776818, 6.5550282, 20.494347, 6.4776818]
flavor_index = {'numucc' : 0 , 'nuecc' : 1, 'nutaucc' : 2, 'NC' : 3}
colors = {'numucc':'green', 'nuecc':'orange', 'nutaucc':'yellow', 'NC':'violet'}
#  data_3view = getData('cvn_vd')
#  data_sim = getData('cvn_sim_vd')
data_gaushit = getData('cvn_gaushit')
data_3view = getData('cvn_hd')
data_sim = getData('cvn_hd', True)


def compareCVN(flavor):
    global data_3view, data_sim, data_gaushit, flavor_index, cvn_vd_pots, cvn_hd_pots
    #  global data_gaushit
    idx = flavor_index[flavor]
  
    bkg_index = {f:flavor_index[f] for f in flavor_index if f != flavor}
    energies_3view = data_3view['test_values']
    true_idx_3view = data_3view['y_test_flavour'][:,0]
    true_pot_3view = np.array([cvn_hd_pots[i] for i in true_idx_3view])
    energies_sim = data_sim['test_values']
    true_idx_sim = data_sim['y_test_flavour'][:,0]
    true_pot_sim = np.array([cvn_hd_pots[i] for i in true_idx_sim])
    energies_gaushit = data_gaushit['test_values']
    true_idx_gaushit = data_gaushit['y_test_flavour'][:,0]
    true_pot_gaushit = np.array([cvn_vd_pots[i] for i in true_idx_gaushit])

    n_3view = len(data_3view['Y_pred'][1][:, idx])
    n_sim = len(data_sim['Y_pred'][1][:, idx])
    n_gaushit = len(data_gaushit['Y_pred'][1][:, idx])
    
    y_pred_sig_3view = data_3view['Y_pred'][1][:, idx][true_idx_3view == idx]
    denom_energies_3view = np.array([val['fNuEnergy'] for val in energies_3view[true_idx_3view == idx]])
    y_pred_sig_sim = data_sim['Y_pred'][1][:, idx][true_idx_sim == idx]
    denom_energies_sim = np.array([val['fNuEnergy'] for val in energies_sim[true_idx_sim == idx]])
    y_pred_sig_gaushit = data_gaushit['Y_pred'][1][:, idx][true_idx_gaushit == idx]
    denom_energies_gaushit = np.array([val['fNuEnergy'] for val in energies_gaushit[true_idx_gaushit == idx]])


    denom_osc_sig_3view = np.array([val['fEventWeight'] for val in energies_3view[true_idx_3view == idx]])/true_pot_3view[true_idx_3view == idx]
    denom_lep_energies_3view = np.array([val['fLepEnergy'] for val in energies_3view[true_idx_3view == idx]])
    denom_lep_frac_3view = denom_lep_energies_3view/denom_energies_3view
    denom_osc_sig_sim = np.array([val['fEventWeight'] for val in energies_sim[true_idx_sim == idx]])/true_pot_sim[true_idx_sim == idx]
    denom_lep_energies_sim = np.array([val['fLepEnergy'] for val in energies_sim[true_idx_sim == idx]])
    denom_lep_frac_sim = denom_lep_energies_sim/denom_energies_sim
    denom_osc_sig_gaushit = np.array([val['fEventWeight'] for val in energies_gaushit[true_idx_gaushit == idx]])/true_pot_gaushit[true_idx_gaushit == idx]
    denom_lep_energies_gaushit = np.array([val['fLepEnergy'] for val in energies_gaushit[true_idx_gaushit == idx]])
    denom_lep_frac_gaushit = denom_lep_energies_gaushit/denom_energies_gaushit
    
    y_pred_bkg_3view = data_3view['Y_pred'][1][:, idx][true_idx_3view != idx]
    y_pred_bkg_sim = data_sim['Y_pred'][1][:, idx][true_idx_sim != idx]
    y_pred_bkg_gaushit = data_gaushit['Y_pred'][1][:, idx][true_idx_gaushit != idx]
    denom_osc_bkg_gaushit = np.array([val['fEventWeight'] for val in energies_gaushit[true_idx_gaushit != idx]])/true_pot_gaushit[true_idx_gaushit != idx]
    y_pred_bkg_comps_gaushit = []
    denom_osc_bkg_comps_gaushit = []
    for f in bkg_index:
        y_pred_bkg_comps_gaushit.append(data_gaushit['Y_pred'][1][:, idx][true_idx_gaushit == bkg_index[f]])
        denom_osc_bkg_comps_gaushit.append(np.array([val['fEventWeight'] for val in energies_gaushit[true_idx_gaushit == bkg_index[f]]]))
    denom_osc_bkg_sim = np.array([val['fEventWeight'] for val in energies_sim[true_idx_sim != idx]])/true_pot_sim[true_idx_sim != idx]
    y_pred_bkg_comps_sim = []
    denom_osc_bkg_comps_sim = []
    for f in bkg_index:
        y_pred_bkg_comps_sim.append(data_sim['Y_pred'][1][:, idx][true_idx_sim == bkg_index[f]])
        denom_osc_bkg_comps_sim.append(np.array([val['fEventWeight'] for val in energies_sim[true_idx_sim == bkg_index[f]]]))
    denom_osc_bkg_3view = np.array([val['fEventWeight'] for val in energies_3view[true_idx_3view != idx]])/true_pot_3view[true_idx_3view != idx]
    y_pred_bkg_comps_3view = []
    denom_osc_bkg_comps_3view = []
    for f in bkg_index:
        y_pred_bkg_comps_3view.append(data_3view['Y_pred'][1][:, idx][true_idx_3view == bkg_index[f]])
        denom_osc_bkg_comps_3view.append(np.array([val['fEventWeight'] for val in energies_3view[true_idx_3view == bkg_index[f]]]))

    b = np.arange(0, 1.02, 0.02, dtype=float)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.title(flavor+'_scores')

    e_sig_gaushit = axs[0].hist(y_pred_sig_gaushit, bins = b, histtype='step', color='red',density=False, linestyle='solid', label = 'vd_sig', weights=denom_osc_sig_gaushit)
    e_bkg_gaushit = axs[0].hist(y_pred_bkg_gaushit, bins = b, histtype='step', color='blue',density=False,linestyle='solid', label = 'vd_bkg', weights=denom_osc_bkg_gaushit)
    e_sig_3view = axs[0].hist(y_pred_sig_3view, bins = b, histtype='step', color='red',density=False, linestyle='dashed', label = 'hd_sig', weights=denom_osc_sig_3view)
    e_bkg_3view = axs[0].hist(y_pred_bkg_3view, bins = b, histtype='step', color='blue',density=False,linestyle='dashed', label = 'hd_bkg', weights=denom_osc_bkg_3view)
    e_sig_sim = axs[0].hist(y_pred_sig_sim, bins = b, histtype='step', color='red',density=False, linestyle='dotted', label = 'hd_old_sig', weights=denom_osc_sig_sim)
    e_bkg_sim = axs[0].hist(y_pred_bkg_sim, bins = b, histtype='step', color='blue',density=False,linestyle='dotted', label = 'hd_old_bkg', weights=denom_osc_bkg_sim)
    #  for i in range(len(y_pred_bkg_comps_gaushit)):
    #      pred_comp = y_pred_bkg_comps_gaushit[i]
    #      osc_comp = denom_osc_bkg_comps_gaushit[i]
    #      key_color = colors[list(bkg_index.keys())[i]]
    #      key_label = list(bkg_index.keys())[i]
    #      axs[0].hist(pred_comp, bins=b, histtype='step',color=key_color,density=False,linestyle='dashed',label=key_label,weights=osc_comp)
    #  for i in range(len(y_pred_bkg_comps_3view)):
    #      pred_comp = y_pred_bkg_comps_3view[i]
    #      osc_comp = denom_osc_bkg_comps_3view[i]
    #      key_color = colors[list(bkg_index.keys())[i]]
    #      key_label = list(bkg_index.keys())[i]
    #      axs[0].hist(pred_comp, bins=b, histtype='step',color=key_color,density=False,linestyle='dashed',label=key_label,weights=osc_comp)
    #  for i in range(len(y_pred_bkg_comps_sim)):
    #      pred_comp = y_pred_bkg_comps_sim[i]
    #      osc_comp = denom_osc_bkg_comps_sim[i]
    #      key_color = colors[list(bkg_index.keys())[i]]
    #      key_label = list(bkg_index.keys())[i]
    #      axs[0].hist(pred_comp, bins=b, histtype='step',color=key_color,density=False,linestyle='dashed',label=key_label,weights=osc_comp)
    
    getN = lambda e : e[0]/np.sum(e[0])
    n_sig_3view = getN(e_sig_3view)
    n_bkg_3view = getN(e_bkg_3view)
    n_sig_sim = getN(e_sig_sim)
    n_bkg_sim = getN(e_bkg_sim)
    n_sig_gaushit = getN(e_sig_gaushit)
    n_bkg_gaushit = getN(e_bkg_gaushit)

    axs[0].set_yscale('log')
    axs[0].set_ylabel('Events/10^23 POT')
    axs[0].set_xlabel('CVN Score')
    axs[0].legend(loc='upper center')

    eff_sig_3view = np.insert(1 - np.cumsum(n_sig_3view), 0, 1.)[:-1]
    eff_bkg_3view = np.insert(1 - np.cumsum(n_bkg_3view), 0, 1.)[:-1]
    pur_3view = eff_sig_3view*np.sum(denom_osc_sig_3view)/(eff_sig_3view*np.sum(denom_osc_sig_3view) + eff_bkg_3view*np.sum(denom_osc_bkg_3view))

    eff_sig_sim = np.insert(1 - np.cumsum(n_sig_sim), 0, 1.)[:-1]
    eff_bkg_sim = np.insert(1 - np.cumsum(n_bkg_sim), 0, 1.)[:-1]
    pur_sim = eff_sig_sim*np.sum(denom_osc_sig_sim)/(eff_sig_sim*np.sum(denom_osc_sig_sim) + eff_bkg_sim*np.sum(denom_osc_bkg_sim))
    
    eff_sig_gaushit = np.insert(1 - np.cumsum(n_sig_gaushit), 0, 1.)[:-1]
    eff_bkg_gaushit = np.insert(1 - np.cumsum(n_bkg_gaushit), 0, 1.)[:-1]
    pur_gaushit = eff_sig_gaushit*np.sum(denom_osc_sig_gaushit)/(eff_sig_gaushit*np.sum(denom_osc_sig_gaushit) + eff_bkg_gaushit*np.sum(denom_osc_bkg_gaushit))
    
    max_gaushit = 100.*max(eff_sig_gaushit*pur_gaushit)
    eff_gaushit_opt = float(100.*eff_sig_gaushit[np.argmax(eff_sig_gaushit*pur_gaushit)])
    pur_gaushit_opt = float(100.*pur_gaushit[np.argmax(eff_sig_gaushit*pur_gaushit)])
    cvn_cut_gaushit = b[np.argmax(eff_sig_gaushit*pur_gaushit)]
    #  if flavor=='nuecc':
    #      max_gaushit=100.*0.447321
    #      cvn_cut_gaushit = 0.82
    #      eff_gaushit_opt=float(100.*eff_sig_gaushit[np.argmax(eff_sig_gaushit*pur_gaushit)+6])
    #      pur_gaushit_opt=float(100.*pur_gaushit[np.argmax(eff_sig_gaushit*pur_gaushit)+6])
    #      print(eff_gaushit_opt*pur_gaushit_opt)
    num_energies_gaushit = denom_energies_gaushit[y_pred_sig_gaushit >= cvn_cut_gaushit]
    num_lep_frac_gaushit = denom_lep_frac_gaushit[y_pred_sig_gaushit >= cvn_cut_gaushit]
    num_osc_sig_gaushit = denom_osc_sig_gaushit[y_pred_sig_gaushit >= cvn_cut_gaushit]
    num_osc_bkg_gaushit = denom_osc_bkg_gaushit[y_pred_bkg_gaushit >= cvn_cut_gaushit]
    for i in range(len(y_pred_bkg_comps_gaushit)):
        pred_comp = y_pred_bkg_comps_gaushit[i]
        osc_comp = denom_osc_bkg_comps_gaushit[i]
        osc_comp = osc_comp[pred_comp >= cvn_cut_gaushit]
        print(list(bkg_index.keys())[i], np.sum(osc_comp)/(np.sum(num_osc_sig_gaushit)+np.sum(num_osc_bkg_gaushit)))

    max_sim = 100.*max(eff_sig_sim*pur_sim)
    eff_sim_opt = float(100.*eff_sig_sim[np.argmax(eff_sig_sim*pur_sim)])
    pur_sim_opt = float(100.*pur_sim[np.argmax(eff_sig_sim*pur_sim)])
    cvn_cut_sim = b[np.argmax(eff_sig_sim*pur_sim)]
    #  if flavor=='nuecc':
    #      max_sim=100.*0.447321
    #      cvn_cut_sim = 0.82
    #      eff_sim_opt=float(100.*eff_sig_sim[np.argmax(eff_sig_sim*pur_sim)+6])
    #      pur_sim_opt=float(100.*pur_sim[np.argmax(eff_sig_sim*pur_sim)+6])
    #      print(eff_sim_opt*pur_sim_opt)
    num_energies_sim = denom_energies_sim[y_pred_sig_sim >= cvn_cut_sim]
    num_lep_frac_sim = denom_lep_frac_sim[y_pred_sig_sim >= cvn_cut_sim]
    num_osc_sig_sim = denom_osc_sig_sim[y_pred_sig_sim >= cvn_cut_sim]
    num_osc_bkg_sim = denom_osc_bkg_sim[y_pred_bkg_sim >= cvn_cut_sim]
    for i in range(len(y_pred_bkg_comps_sim)):
        pred_comp = y_pred_bkg_comps_sim[i]
        osc_comp = denom_osc_bkg_comps_sim[i]
        osc_comp = osc_comp[pred_comp >= cvn_cut_sim]
        print(list(bkg_index.keys())[i], np.sum(osc_comp)/(np.sum(num_osc_sig_sim)+np.sum(num_osc_bkg_sim)))
    
    
    max_3view = 100.*max(eff_sig_3view*pur_3view)
    eff_3view_opt = float(100.*eff_sig_3view[np.argmax(eff_sig_3view*pur_3view)])
    pur_3view_opt = float(100.*pur_3view[np.argmax(eff_sig_3view*pur_3view)])
    cvn_cut_3view = b[np.argmax(eff_sig_3view*pur_3view)]
    #  if flavor=='nuecc':
    #      max_3view=100.*0.447321
    #      cvn_cut_3view = 0.82
    #      eff_3view_opt=float(100.*eff_sig_3view[np.argmax(eff_sig_3view*pur_3view)+6])
    #      pur_3view_opt=float(100.*pur_3view[np.argmax(eff_sig_3view*pur_3view)+6])
    #      print(eff_3view_opt*pur_3view_opt)
    num_energies_3view = denom_energies_3view[y_pred_sig_3view >= cvn_cut_3view]
    num_lep_frac_3view = denom_lep_frac_3view[y_pred_sig_3view >= cvn_cut_3view]
    num_osc_sig_3view = denom_osc_sig_3view[y_pred_sig_3view >= cvn_cut_3view]
    num_osc_bkg_3view = denom_osc_bkg_3view[y_pred_bkg_3view >= cvn_cut_3view]
    for i in range(len(y_pred_bkg_comps_3view)):
        pred_comp = y_pred_bkg_comps_3view[i]
        osc_comp = denom_osc_bkg_comps_3view[i]
        osc_comp = osc_comp[pred_comp >= cvn_cut_3view]
        print(list(bkg_index.keys())[i], np.sum(osc_comp)/(np.sum(num_osc_sig_3view)+np.sum(num_osc_bkg_3view)))
    
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    axs[1].plot(b[:-1], eff_sig_3view*pur_3view, color='violet', label='HD, max fom : %.02f'% max_3view)
    axs[1].plot(b[:-1], eff_sig_sim*pur_sim, color='green', label='HD old, max fom : %.02f' % max_sim)
    axs[1].plot(b[:-1], eff_sig_gaushit*pur_gaushit, color='red', label='VD, max fom : %.02f' % max_gaushit)
    axs[1].set_ylabel('Efficiency x Purity')
    axs[1].set_xlabel('CVN Score')
    axs[1].legend(loc='lower center')

    plt.savefig('plots_8x6/'+flavor+'_cvn_dist.pdf')

    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    plt.title(flavor+' selection efficiency')
    b = np.arange(0, 10, 0.5, dtype=float)
    
    h_denom_gaushit = axs[0].hist(denom_energies_gaushit, bins = b, histtype='step', color='red', label = 'all VD '+flavor, linestyle = 'solid',weights=denom_osc_sig_gaushit) 
    h_denom_sim = axs[0].hist(denom_energies_sim, bins = b, histtype='step', color='red', label = 'all HD old '+flavor, linestyle = 'dotted',weights=denom_osc_sig_sim) 
    h_denom_3view = axs[0].hist(denom_energies_3view, bins = b, histtype='step', color='red', label = 'all HD '+flavor, linestyle = 'dashed',weights=denom_osc_sig_3view) 
    h_num_gaushit = axs[0].hist(num_energies_gaushit, bins = b, histtype='step', color='blue', linestyle='solid', label = 'selected VD '+flavor,weights=num_osc_sig_gaushit) 
    h_num_3view = axs[0].hist(num_energies_3view, bins = b, histtype='step', color='blue', linestyle='dotted', label = 'selected HD '+flavor,weights=num_osc_sig_3view) 
    h_num_sim = axs[0].hist(num_energies_sim, bins = b, histtype='step', color='blue', linestyle='dashed', label = 'selected HD old '+flavor,weights=num_osc_sig_sim) 
    axs[0].set_ylabel('Number of Events')
    axs[0].set_xlabel('True Energy (GeV)')
    axs[0].legend(loc='upper right')
   
    axs[1].set_ylabel('Selection Efficiency')
    axs[1].set_xlabel('True Energy (GeV)')
    axs[1].set_ylim([0., 1.])
    axs[1].plot(b[:-1], np.divide(h_num_3view[0], h_denom_3view[0], where=(h_denom_3view[0] != 0)), color='violet', label = 'HD')
    axs[1].plot(b[:-1], np.divide(h_num_sim[0], h_denom_sim[0], where=(h_denom_sim[0] != 0)), color='green', label = 'HD Old')
    axs[1].plot(b[:-1], np.divide(h_num_gaushit[0], h_denom_gaushit[0], where=(h_denom_gaushit[0] != 0)), color='red', label = 'VD')
    axs[1].legend(loc = 'lower right')
    axs[1].text(0., 0.45, 'Overall Efficiencies (HD, HD Old, VD) : %.02f %%, %.02f %%, %.02f %%'% (eff_3view_opt, eff_sim_opt, eff_gaushit_opt), fontsize=7)
    axs[1].text(0., 0.4, 'Overall Purities : %.02f %%, %.02f %%, %.02f %%'% (pur_3view_opt, pur_sim_opt, pur_gaushit_opt), fontsize=7)
    axs[1].text(0., 0.35, 'CVN Cuts : %.02f, %.02f, %.02f'%(cvn_cut_3view, cvn_cut_sim, cvn_cut_gaushit), fontsize=7)
    #  axs[1].text(0., 0.45, 'Overall Efficiencies (gaushit) : %.02f %%'% (eff_gaushit_opt), fontsize=7)
    #  axs[1].text(0., 0.4, 'Overall Purities : %.02f %%'% (pur_gaushit_opt), fontsize=7)
    print("Purities, efficiencies : ", pur_gaushit_opt, eff_gaushit_opt)
    #  axs[1].text(0., 0.35, 'CVN Cuts : %.02f'%(cvn_cut_gaushit), fontsize=7)
    
    plt.savefig('plots_8x6/'+flavor+'_cvn_sel_eff_energy.pdf')
    
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    plt.title(flavor+' selection efficiency')
    b = np.arange(0, 1, 0.1, dtype=float)
    
    h_denom_gaushit = axs[0].hist(denom_lep_frac_gaushit, bins = b, histtype='step', color='red', label = 'all VD '+flavor, linestyle = 'solid',weights=denom_osc_sig_gaushit) 
    h_denom_sim = axs[0].hist(denom_lep_frac_sim, bins = b, histtype='step', color='red', label = 'all HD old '+flavor, linestyle = 'dotted',weights=denom_osc_sig_sim) 
    h_denom_3view = axs[0].hist(denom_lep_frac_3view, bins = b, histtype='step', color='red', label = 'all HD '+flavor, linestyle = 'dashed',weights=denom_osc_sig_3view) 
    h_num_gaushit = axs[0].hist(num_lep_frac_gaushit, bins = b, histtype='step', color='blue', linestyle='solid', label = 'selected VD '+flavor,weights=num_osc_sig_gaushit) 
    h_num_3view = axs[0].hist(num_lep_frac_3view, bins = b, histtype='step', color='blue', linestyle='dotted', label = 'selected HD '+flavor,weights=num_osc_sig_3view) 
    h_num_sim = axs[0].hist(num_lep_frac_sim, bins = b, histtype='step', color='blue', linestyle='dashed', label = 'selected HD old '+flavor,weights=num_osc_sig_sim) 
    axs[0].set_ylabel('Number of Events')
    axs[0].set_xlabel('Leptonic Energy Fraction')
    axs[0].legend(loc='upper right')
   
    axs[1].set_ylabel('Selection Efficiency')
    axs[1].set_xlabel('Lepton Energy Fraction')
    axs[1].set_ylim([0., 1.])
    axs[1].plot(b[:-1], np.divide(h_num_3view[0], h_denom_3view[0], where=(h_denom_3view[0] != 0)), color='violet', label = 'HD')
    axs[1].plot(b[:-1], np.divide(h_num_sim[0], h_denom_sim[0], where=(h_denom_sim[0] != 0)), color='green', label = 'HD Old')
    axs[1].plot(b[:-1], np.divide(h_num_gaushit[0], h_denom_gaushit[0], where=(h_denom_gaushit[0] != 0)), color='red', label = 'VD')
    axs[1].legend(loc = 'lower right')
    axs[1].text(0., 0.45, 'Overall Efficiencies (HD, HD Old, VD) : %.02f %%, %.02f %%, %.02f %%'% (eff_3view_opt, eff_sim_opt, eff_gaushit_opt), fontsize=7)
    axs[1].text(0., 0.4, 'Overall Purities : %.02f %%, %.02f %%, %.02f %%'% (pur_3view_opt, pur_sim_opt, pur_gaushit_opt), fontsize=7)
    axs[1].text(0., 0.35, 'CVN Cuts : %.02f, %.02f, %.02f'%(cvn_cut_3view, cvn_cut_sim, cvn_cut_gaushit), fontsize=7)
    #  axs[1].text(0., 0.45, 'Overall Efficiencies (gaushit) : %.02f %%'% (eff_gaushit_opt), fontsize=7)
    #  axs[1].text(0., 0.4, 'Overall Purities : %.02f %%'% (pur_gaushit_opt), fontsize=7)
    print("Purities, efficiencies : ", pur_gaushit_opt, eff_gaushit_opt)
    #  axs[1].text(0., 0.35, 'CVN Cuts : %.02f'%(cvn_cut_gaushit), fontsize=7)
    
    plt.savefig('plots_8x6/'+flavor+'_cvn_sel_eff_lepfrac.pdf')


compareCVN('nuecc')
compareCVN('numucc')
compareCVN('NC')

#  def get_info(key, variant):
#      global path
#      with open(path+'/'+variant+'/3view/'+key+'.info', 'rb') as info_file:
#          vals = info_file.readlines()
#          return float(vals[1]), int(vals[7])
#
#  def get_pred(key, data, flavor, ids, verbose=False):
#      if key not in ids:
#          return
#      return data['Y_pred'][1][:, flavor][ids == key][0]
#
#  import zlib
#  def get_pixelmap(key, variant):
#      global path
#      with open(path+'/'+variant+'/3view/'+key+'.gz', 'rb') as image_file:
#          pixels = np.fromstring(zlib.decompress(image_file.read()), dtype=np.uint8, sep='').reshape(3, 500, 500)
#          pixels = pixels.astype('float32')
#          pixels /= 255.
#          return pixels
#
#  def get_pminfo(pm, view=None):
#      npixels = len(pm[pm!= 0.])
#      nsum = np.sum(pm)
#      nstd = np.std(pm[pm!= 0.])
#      if view is not None:
#          npixels = len(pm[view][pm[view]!= 0.])
#          nsum = np.sum(pm[view])
#          nstd = np.std(pm[view][pm[view]!= 0.])
#      return npixels, nsum, nstd
#
#  def draw_pm(pm1, pm2, energy, pred1, pred2, key):
#      fig, axs = plt.subplots(2, 3, figsize=(15, 10))
#      fig.suptitle('Energy : %.02f, cvn wire : %.02f, cvn gaushit : %.02f'%(energy,pred1, pred2))
#      titles = ['U', 'V', 'Z']
#      for i in range(3):
#          maps = np.flip(np.swapaxes(pm[i], 0, 1), axis=0)
#          maps2 = np.flip(np.swapaxes(pm2[i], 0, 1), axis=0)
#          axs[0, i].imshow(maps, interpolation='none')
#          axs[0, i].set_xlabel('Wire')
#          axs[0, i].set_ylabel('TDC')
#          axs[0, i].title.set_text(titles[i])
#          axs[1, i].imshow(maps2, interpolation='none')
#          axs[1, i].set_xlabel('Wire')
#          axs[1, i].set_ylabel('TDC')
#      name = key.split('/')[3]
#      plt.savefig('pm_plots_gaushit/'+name+'_pm.pdf')
#
#  def draw_single_pm(pm1, energy, pred1, key,t='sig'):
#      fig, axs = plt.subplots(1, 3, figsize=(15, 10))
#      fig.suptitle('Energy : %.02f, cvn gaushit : %.02f'%(energy,pred1))
#      titles = ['U', 'V', 'Z']
#      for i in range(3):
#          maps = np.flip(np.swapaxes(pm1[i], 0, 1), axis=0)
#          axs[i].imshow(maps, interpolation='none')
#          axs[i].set_xlabel('Wire')
#          axs[i].set_ylabel('TDC')
#          axs[i].title.set_text(titles[i])
#      name = key.split('/')[3]
#      plt.savefig('pm_plots_gaushit_8x6/'+t+'_'+name+'_pm.pdf')
#  #
#  count = 0
#  count2 = 0
#  count3 = 0
#  #  labels, partition = getIDs('cvn_vd')
#  #  labels_sim, partition_sim = getIDs('cvn_sim_vd')
#  labels_gaushit, partition_gaushit = getIDs('cvn_gaushit')
#  #
#  #  #  keys = sorted(list(partition['test']))
#  #  #  keys_sim = sorted(list(partition_sim['test']))
#  keys_gaushit = sorted(list(partition_gaushit['test']))
#  #  keys = sorted(list(partition['test']))
#  #  keys_sim = sorted(list(partition_sim['test']))
#  #
#  #  diff = np.array([], dtype=float)
#  #  diff2 = np.array([], dtype=float)
#  #  diff3 = np.array([], dtype=float)
#  #  ids = np.array([y['ID'] for y in data_3view['test_values']])
#  #  ids2 = np.array([y['ID'] for y in data_sim['test_values']])
#  ids3 = np.array([y['ID'] for y in data_gaushit['test_values']])
#  #  npixels, nsum, nstd = np.array([]), np.array([]), np.array([])
#  #  npixels2, nsum2, nstd2 = np.array([]), np.array([]), np.array([])
#  #  npixels3, nsum3, nstd3 = np.array([]), np.array([]), np.array([])
#  #  print(len(keys), len(keys_sim), len(keys_gaushit))
#  for i in range(len(keys_gaushit)):
#      #  key = keys[i]
#      #  key_sim = keys_sim[i]
#      key_gaushit = keys_gaushit[i]
#      #  info = get_info(key, 'cvn_vd')
#      #  info2 = get_info(key_sim, 'cvn_sim_vd')
#      info3 = get_info(key_gaushit, 'cvn_gaushit')
#      #  assert(info == info3)
#      #  pm = get_pixelmap(key, 'cvn_vd')
#      #  pm2 = get_pixelmap(key_sim, 'cvn_sim_vd')
#      pm3 = get_pixelmap(key_gaushit, 'cvn_gaushit')
#      #  pred = get_pred(key, data_3view, 1, ids)
#      #  if not pred: continue
#      #  pred2 = get_pred(key_sim, data_sim, 1, ids2)
#      #  if not pred2: continue
#      pred3 = get_pred(key_gaushit, data_gaushit, 1, ids3)
#      if not pred3: continue
#      #  if(info[1] == 12):
#      #      temp_x, temp_y, temp_z = [], [], []
#      #      for j in [None, 0, 1, 2]:
#      #          x, y, z= get_pminfo(pm, j)
#      #          temp_x.append(x)
#      #          temp_y.append(y)
#      #          temp_z.append(z)
#      #      npixels = np.append(npixels, temp_x)
#      #      nsum = np.append(nsum, temp_y)
#      #      nstd = np.append(nstd, temp_z)
#      #  if(info2[1] == 12):
#      #      temp_x2, temp_y2, temp_z2 = [], [], []
#      #      for j in [None, 0, 1, 2]:
#      #          x2, y2, z2= get_pminfo(pm2, j)
#      #          temp_x2.append(x2)
#      #          temp_y2.append(y2)
#      #          temp_z2.append(z2)
#      #      npixels2 = np.append(npixels2, temp_x2)
#      #      nsum2 = np.append(nsum2, temp_y2)
#      #      nstd2 = np.append(nstd2, temp_z2)
#      #  if(info3[1] == 12):
#      #      temp_x3, temp_y3, temp_z3 = [], [], []
#      #      for j in [None, 0, 1, 2]:
#      #          x3, y3, z3= get_pminfo(pm3, j)
#      #          temp_x3.append(x3)
#      #          temp_y3.append(y3)
#      #          temp_z3.append(z3)
#      #      npixels3 = np.append(npixels3, temp_x3)
#      #      nsum3 = np.append(nsum3, temp_y3)
#      #      nstd3 = np.append(nstd3, temp_z3)
#
#      if(info3[1] == 12) and (pred3 >= 0.82) and (info3[0] > 2) and (info3[0] < 6):
#          count+=1
#          if count > 100: break
#          draw_single_pm(pm3, info3[0], pred3, key_gaushit)
#      if(info3[1]==16) and (pred3 >= 0.82) and (info3[0] > 2) and (info3[0] < 6):
#          count2+=1
#          if count2 > 25: break
#          draw_single_pm(pm3, info3[0], pred3, key_gaushit,'nutaucc')
#      if(info3[1]==1) and (pred3 >= 0.82) and (info3[0] > 2) and (info3[0] < 6):
#          count3+=1
#          if count3 > 50: break
#          draw_single_pm(pm3, info3[0], pred3, key_gaushit,'NC')
#
#
#
#      #  if (info[0] <= 0.5) and (info[1] == 12):
#      #      #  diff = np.append(diff, pred2-pred)
#      #      count += 1
#      #      if count > 100: break
#      #      draw_pm(pm, pm3, info[0], pred, pred3, key)
#      #  if (info[0] >= 0.5) and (info[0] <= 1.5) and (info[1] == 12):
#      #      #  diff = np.append(diff, pred2-pred)
#      #      count2 += 1
#      #      if count2 > 100: break
#      #      draw_pm(pm, pm3, info[0], pred, pred3, key)
#
#      #  if (info[0] > 0.5) and (info[0] <= 1) and (info[1] == 12):
#      #      diff2 = np.append(diff2, pred2-pred)
#      #  if (info[0] > 1) and (info[1] == 12):
#      #      diff3 = np.append(diff3, pred2-pred)
#  #
#  #  npixels = npixels.reshape(-1, 4)
#  #  nsum = nsum.reshape(-1, 4)
#  #  nstd = nstd.reshape(-1, 4)
#  #  npixels2 = npixels2.reshape(-1, 4)
#  #  nsum2 = nsum2.reshape(-1, 4)
#  #  nstd2 = nstd2.reshape(-1, 4)
#  #
#  #  npixels3 = npixels3.reshape(-1, 4)
#  #  nsum3 = nsum3.reshape(-1, 4)
#  #  nstd3 = nstd3.reshape(-1, 4)
#  #  #  fig, ax = plt.subplots(1, 1, figsize=(5,5))
#  #  #  ax.hist(diff, bins=20, histtype='step', color='red', label='energy < 0.5, mean : %.02f'%np.mean(diff))
#  #  #  ax.hist(diff2, bins=20, histtype='step', color='blue', label='0.5 < energy < 1, mean : %.02f'%np.mean(diff2))
#  #  #  ax.hist(diff3, bins=20, histtype='step', color='green', label='energy > 1, mean : %.02f' %np.mean(diff3))
#  #  #  ax.title.set_text('cvn sim - cvn wire (nuecc)')
#  #  #  ax.legend(loc='best')
#  #  #  plt.savefig('plots/deltacvn_nuecc_simwire.pdf')
#  #
#  #  def compare_pminfo(arr1, arr2, arr3, title, fname):
#  #      views = ['Total', 'U', 'V', 'Z']
#  #      fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#  #      fig.suptitle(title)
#  #      for i in range(len(views)):
#  #          x = i // 2
#  #          y = i % 2
#  #          axs[x, y].hist(arr1[:,i], bins=25, histtype='step', color='red', label='wire')
#  #          axs[x, y].hist(arr2[:,i], bins=25, histtype='step', color='blue', label='sim channel')
#  #          axs[x, y].hist(arr3[:,i], bins=25, histtype='step', color='green', label='gaushit')
#  #          axs[x, y].legend(loc='best')
#  #          axs[x, y].title.set_text(views[i])
#  #      plt.savefig('plots/'+fname+'.pdf')
#  #
#  #      #  fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#  #      #  fig.suptitle('Diff : '+title)
#  #      #  for i in range(len(views)):
#  #      #      x = i // 2
#  #      #      y = i % 2
#  #      #      axs[x, y].hist(arr2[:,i]-arr1[:,i], bins=25, histtype='step', color='red', label='sim-wire')
#  #      #      axs[x, y].hist(arr2[:,i]-arr3[:,i], bins=25, histtype='step', color='blue', label='sim-gaushit')
#  #      #      axs[x, y].title.set_text(views[i])
#  #      #  plt.savefig('plots/diff_'+fname+'.pdf')
#  #
#  #  #  compare_pminfo(npixels, npixels2, npixels3, 'Number of Pixels', 'nuecc_npixels')
#  #  #  compare_pminfo(nsum, nsum2, nsum3, 'Total Charge', 'nuecc_nsum')
#  #  #  compare_pminfo(nstd, nstd2, nstd3, 'Standard Deviation of Charge', 'nuecc_nstd')
