function proc_plotERP(epo, ival)

epo_rSquare=proc_rSquareSigned(epo ,'Stats',1);
mnt = mnt_setElectrodePositions(epo.clab);
figure('Units','points','position',[200, 200, 900, 600])
epo_rSquare.className={'sgn r^2 (t, nt)'};
epo.className={'t','nt'};
plot_scalpEvolutionPlusChannelPlusrSquared(epo,epo_rSquare, mnt,{'Cz','CP5'}, ival,'Resolution',50,...
    'GlobalCLim',[],'RsquareColormap',cmap_posneg(51),...
   'Colormap',jet,'ColorOrder', [0 0 1;0 0.5 0],'extrapolateToZero',1);

end
