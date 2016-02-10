cfg                         = [];
cfg.dataset                 = 'P1.bdf';
cfg.trialfun                = 'ft_trialfun_general'; % this is the default
cfg.trialdef.eventtype      = 'STATUS';
cfg.trialdef.eventvalue     = 243; % the value of the stimulus trigger for fully incongruent (FIC).
cfg.trialdef.prestim        = 2; % in seconds
cfg.trialdef.poststim       = 2; % in seconds
cfg = ft_definetrial(cfg);
cfg.continuous = 'yes';

vol_action = ft_preprocessing(cfg);

%%
cfg                         = [];
cfg.dataset                 = 'P1.bdf';
cfg.trialfun                = 'ft_trialfun_general'; % this is the default
cfg.trialdef.eventtype      = 'STATUS';
cfg.trialdef.eventvalue     = 219; % the value of the stimulus trigger for fully incongruent (FIC).
cfg.trialdef.prestim        = 2; % in seconds
cfg.trialdef.poststim       = 2; % in seconds
cfg = ft_definetrial(cfg);
cfg.continuous = 'yes';

inv_action = ft_preprocessing(cfg);

%%

cfg = [];
cfg.reref           = 'yes';
cfg.refchannel      = 'all';
cfg.refmethod       = 'avg';
cfg.lpfilter        = 'yes';
cfg.lpfreq          = 90;
% cfg.hpfilter        = 'yes';
% cfg.hpfreq          = 1.5;
% cfg.bpfilter        = 'yes';
% cfg.bpfreq          = [1, 90];
cfg.demean          = 'yes';
cfg.baselinewindow  = [-2,  -1.5];

vol_action_pre      = ft_preprocessing(cfg, vol_action);
inv_action_pre      = ft_preprocessing(cfg, inv_action);


%%
cfg = []
cfg.channel = 'EEG';
cfg.layout  = 'biosemi64.lay'

ft_multiplotER(cfg, tf_vol, tf_inv)

%%

cfg = [];

vol_ave = ft_timelockanalysis(cfg, vol_action_pre)
inv_ave = ft_timelockanalysis(cfg, inv_action_pre)


%%
cfg              = [];
cfg.output       = 'pow';
cfg.channel      = 'EEG';
cfg.method       = 'mtmfft';
cfg.foi          = 2:1:48;                         % analysis 2 to 30 Hz in steps of 2 Hz           % time window "slides" from -0.5 to 1.5 sec in steps of 0.05 sec (50 ms)
cfg.tapsmofrq    = 2;
tf_vol = ft_freqanalysis(cfg, vol_action_pre);
tf_inv = ft_freqanalysis(cfg, inv_action_pre);
