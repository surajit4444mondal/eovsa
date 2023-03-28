import sys
import os
sys.path.append('/home/user/test_svn/python/')
sys.path.append('/common/python/')
sys.path.append('/common/python/current/')
sys.path.append('/common/python/packages/pipeline')
sys.path.append('/common/python/packages/pipeline_casa')
sys.path.append('/home/user/project/eovsapy/eovsapy')
sys.path.append(os.getcwd())
import dbutil as db
import numpy as np
from util import Time, nearest_val_idx, common_val_idx, lobe, bl2ord, get_idbdir
import stateframe
import cal_header as ch
import glob
import read_idb as ri
from xspfits2 import tp_writefits
import pipeline_cal as callib
import matplotlib.pyplot as plt

class flare_detector_and_plotter:
    def __init__(self,trange,outpath='./'):
        '''
            trange: This is a 2-element list of Time objects or list of IDB files
            udb_corr: Will run udb_corr if set to True
            thresh: Threshold to determine a probable flare. Ths is in terms of mad
                and is used on the smoothed and normalised lightcurve at each frequency
            min_flux_density: This is the minimum flux that the peak inside a single IDB file
                          should have for it to be considered as related to a flare. The
                          units of this is SFU. However I feel that this is not needed.
            max_diff_same_flare: The maximum time difference for two points to be associated
                             with a single flare. The units are in seconds
            tol=0.2: 20 percent difference from mean power of neighbours
                           
            min_flare_freq=40  #### if a flare is detected at this many frequencies at least
                          ### I take that this is a true flare
            lcfreqs   The list of frequency indexes for lightcurves in the lower plot. Default is [25, 235].
            name      The output filename (stem only, no extension).  If None (default), no plot or
                          binary file is produced.  For production purposes, use the standard naming 
                          convention as follows:
                            name='EOVSA_yyyymmdd_Xflare' where yyyy is year, mm is month, dd is day, and 
                            X is the GOES class.
            time_diff: This much time will be used before and after the detected flare duration during
                       plotting the dynamic spectrum. Unit is seconds
        '''
        
        if type(trange[0])==str:
            self.files=trange
            self.trange=None
        else:
	    self.files=None
            self.trange=trange
	self.udb_corr=True
        if outpath[-1]!='/':  #### small modification for folder
           outpath+='/' 
        self.outpath=outpath
	self.thresh=5.0
	self.min_flux_density=5
	self.max_diff_same_flare=300
	self.min_flare_freq=100
	self.tol=0.2
	self.lcfreqs=[25,100,180,250,390]
	self.name=None
	self.time_diff=300
	self.num_freqs=None
	self.times=None   ### This will contain all times
	self.freqs=None   ### this will contains frequencies in GHz
	self.spec_for_plotting=None
        self.flare_list=None
        self.flare_list_original=None
    
	
    def file_finder_and_calibrator(self):
        '''
            if trange is a list of filenames, I do nothing. If it is a 2-element list of Time objects
            I first search for the relevant IDB files. Then I run udb_crr, which then writes the 
            corrected files int ooutpath. After this step the units in the IDB files is SFU. I also assume
            that the outpath is empty of any IDB files before this code is run. Then I search for all files
            which starts with IDB. I sort the filenames, which ensures that during flare detection, earlier
            times are processed earlier. I also set udb_corr=False, so that it is not run later.
        '''
        if self.files!=None:
            return
        callib.allday_udb_corr(self.trange,self.outpath)
        self.udb_corr=False
        import glob
        files=glob.glob(self.outpath+"IDB*")   
        files.sort()
        self.files=files
        return

    def detect_flare(self):
        '''
              This function is the main workhorse of this Class. It reads each UDB files and calculates all
              quantitites necessary for flare detection and also for the plotting routines which can be
              invoked later. The basic idea behind the flare detection is as follows:
              1. Take the lightcurve at each frequency and then calculate the normalised lightcurve.
                 Normalisation is done using (value-median)/median
              2. Smooth the normalised lightcurve a bit to remove small duration peaks. I use a 5s running mean.
              3. Next get the median absolute deviation of the smoothed lightcurve.
              4. Take the residual between the smoothed and normalised lightcurves and search for positions
                 where the residual is greater than threshold x mad
              5. If such a position exists, I take that as a flare and calculate its start and end.
              6. If not, I do the same, but with threshold=threshold-2. This is done because it is possible that
                 the flare is not detected because the median has been biased by the flare itself, which means
                 that the median, which is supposedly the background is artifically high. In this circumstance,
                 taking a smaller threshold will help. This lower threshold will only be used if the median in
                 this timerange is significantly higher than the neighbouring IDB files.
              7. After this is done for all frequenies, I compare the neighboring IDB files, to see if median at
                 any one is much higher than the neigbors and use that to try and find flares.
              8. I also enforce the condition that if it is a true flare, it should be detected in at least N
                 number of frequencies, where N=min_flare_freq
              9. Then I combine the detected flare based on the start and end times  detected. If the start/end of
                 two flares differs by max_diff_same_flare, then I put then under the same flare.

        '''
        if self.flare_list_original!=None:
            final_flares=self.combine_flares(self.flare_list_original)
            self.flare_list=final_flares
            return
        num_files=len(self.files) 
        obs_times=[None]*num_files
        num_freqs=0
        spec_for_plotting=[]
        for file_idx, file2 in enumerate(self.files):
            if self.udb_corr==True:   #### This is needed only if the user has copied the IDB files from relevant
                                      #### folder, but has not run the udb_corr
                file1=callib.udb_corr(file2,calibrate=True)
            else:
                file1=file2
            out = ri.read_idb([file1])
            nant,npol,nf,nt = out['p'].shape
            ##### Initialising the array for flare finding
            if file_idx==0:
                peak_pos=np.zeros((num_files,nf),dtype=int)*np.nan  ### while I set the arry type to int, nan is a float.
                                                                    ### so the array type becomes float
                low_pos_5sigma=np.zeros((num_files,nf),dtype=int)*np.nan
                high_pos_5sigma=np.zeros((num_files,nf),dtype=int)*np.nan
                low_pos_3sigma=np.zeros((num_files,nf),dtype=int)*np.nan
                high_pos_3sigma=np.zeros((num_files,nf),dtype=int)*np.nan
                flare_start_pos=np.zeros((num_files,nf),dtype=int)*np.nan
                flare_end_pos=np.zeros((num_files,nf),dtype=int)*np.nan
                flare_detected=np.zeros((num_files,nf),dtype=bool)
                peak_power=np.zeros((num_files,nf))
                mean_power=np.zeros((num_files,nf))
                self.num_freqs=nf
                self.freqs=out['fghz']
            out['x']=np.abs(out['x'])
            out['p']=np.abs(out['p'])
            obs_times[file_idx]=out['time']
            nant = 13
            # Use only data from tracking antennas
            azeldict = callib.get_sql_info(Time(out['time'],format='jd')[[0,-1]])
            idx = nearest_val_idx(out['time'],azeldict['Time'].jd)
            tracking = azeldict['TrackFlag'].T
            # Flag any data where the antennas are not tracking
            for i in range(nant):
                out['p'][i,:,:,~tracking[i,idx]] = np.nan
           
            out['p'][nant:,...]=np.nan
            corr_ant_size=15
            ### Here I am flagging the baselines of the antennas which are not tracking.
            ### I also set the auto-correlations to nan. Those numbers are anyway available
            ### from out['p']
            for i in range(corr_ant_size):
                for j in range(i,corr_ant_size):
                    ########################################################################################
                    #### The structure out['x'] contains both the auto-correlations and cross-correlations
                    #### Hence for baseline (0,i) the id is i
                    #### For baseline (1,i) the id is 15+i-1; 15 stands for the total antenna number in correlator
                    ####                                      -1 is present because baseline (1,0) is not stored
                    #### For baseline (2,i), the id is 15+14+i-2
                    ########################################################################################
                    baseline_id=corr_ant_size*(corr_ant_size+1)/2-(corr_ant_size-i)*(corr_ant_size-i+1)/2+j-i
                    if i==j:
                        out['x'][baseline_id,...]=np.nan
                    elif i<nant and j<nant:
                        out['x'][baseline_id,:,:,~tracking[i,idx]]=np.nan  
                        out['x'][baseline_id,:,:,~tracking[j,idx]]=np.nan
                    else:
                        out['x'][baseline_id,...]=np.nan



            ## Taking median over time for all baselines first
            med=np.nanmedian(out['x'],axis=3)  ### shape is now  nbaselines x nPol x nchans
            med=np.expand_dims(med,3)  ### shape is now nbaselines x nPol x nchans x 1
            med_power=np.abs(np.nanmedian(out['x'],axis=(0,1)))
            normalised_power=np.abs(np.nanmedian((out['x']-med)/med,axis=(0,1)))  #### TODO Do we want to take average of
                                                                                  #### both polarisations?
            median_norm_power=np.nanmedian(normalised_power,axis=1)


            #### The lines below, till where I defined med_tot_power
            ### are adapted from Dale Gary's function named allday_process inside pipeline_cal.py
            medt=np.nanmedian(np.nanmedian(out['p'][:nant],3 ),1)  ##size nf,nant
            medspec=np.nanmedian(medt,0)  ## size nf
            p=np.polyfit(out['fghz'],medspec,2)
            model_spec=np.polyval(p,out['fghz']).repeat(nant).reshape(nf,nant)  ### size nf,nant
            stddev=np.nanstd(medt-np.transpose(model_spec),1)
            ant_idx=stddev.argsort()[:8]  ## List of 8 best-fitting antennas
            med_tot_power=np.nanmedian(np.nanmedian(out['p'][ant_idx],0),0)
            #### I do not use the median subtracted total power because the median can be highly biased by the flare.
            #### Hence it is possible that the background subtracted total power becomes very small and is rejected
            #### due to min_flux_density criteria. This is expected to be more relevant for the brighter and longer
            #### duration flare, which in general should be easier to detect. However this essentially means that
            #### right now this is not being used, due to the low min_flux_density used. Maybe we can remove this
            #### check later, once we get more experience running this code.
     
            for i in range(nf):
                peak=0
                smoothed=np.convolve(normalised_power[i,:],np.ones(5),mode='same')*1./5  #### smoothing a bit to take care of small 
                                                                                         #### spurious peaks
                max_pow=np.nanmax(smoothed)
                mad=np.nanmedian(abs(normalised_power[i,:]-median_norm_power[i]))
                pos=np.where(np.isnan(smoothed)==True)[0]
                smoothed[pos]=0.0
                normalised_power[i,pos]=0.0
                y=(smoothed-median_norm_power[i])/mad
                pos=np.argmax(y) 
                peak_power[file_idx,i]=med_tot_power[i,pos]  #### in SFU
                mean_power[file_idx,i]=np.nanmedian(med_power[i,:])
                thresh=self.thresh
                if y[pos]>thresh:
                    peak_pos[file_idx,i]=pos
                    flare_detected[file_idx,i]=True    
                    pos=np.where(y>thresh)[0]
                    start,end=self.get_start_end_flare(peak_pos[file_idx,i],y,thresh,nt)
                    low_pos_5sigma[file_idx,i]=start
                    high_pos_5sigma[file_idx,i]=end
                    start,end=self.get_start_end_flare(peak_pos[file_idx,i],y,1,nt) #### I put start and end, when the flare flux hits median
                    flare_start_pos[file_idx,i]=start
                    flare_end_pos[file_idx,i]=end
                else:
                    #### Try to search for flare with a smaller threshold. This will become important if the flare is 
                    ### so long that it biases the median itself. The idea is that if we detect a flare with this lower threshold
                    ### and the median for this file is significantly higher than the median of its neighbours, then it implies a
                    ### flare for this IDB file.
                    thresh=thresh-2  
                    peak_pos[file_idx,i]=pos
                    flare_detected[file_idx,i]=False
                    pos=np.where(y>thresh)[0]
                    start,end=self.get_start_end_flare(peak_pos[file_idx,i],y,thresh,nt)
                    low_pos_3sigma[file_idx,i]=start
                    high_pos_3sigma[file_idx,i]=end
                    start,end=self.get_start_end_flare(peak_pos[file_idx,i],y,1,nt)
                    flare_start_pos[file_idx,i]=start
                    flare_end_pos[file_idx,i]=end
            spec_for_plotting.append(self.inspect(out,ant_idx)) 
        self.times=Time(np.concatenate(obs_times),format='jd',scale='utc')       
        self.spec_for_plotting=np.concatenate(spec_for_plotting,axis=1)
        possible_flare=np.zeros(num_freqs,dtype=bool)

        for file_idx in range(num_files):
            possible_flare[:]=False
            pos=np.where(flare_detected[file_idx,:]==True)[0]
            if len(pos)>=self.min_flare_freq:  #### checking if flare has been detected at minimum required 
                                          ### number of frequencies
                continue
            #### Here I try to see if the median is biased by the flare.
            #### The assumption here is that the flare duration and strength cannot be such that it biases the
            ### median significantly for a time greater than the duration of the IDB file.
            for i in range(num_freqs):
                current_mean_power=mean_power[file_idx,i]
                count=0
                try:
                    prev_mean_power=mean_power[file_idx-1,i]
                    count+=1
                except IndexError:
                    prev_mean_power=0
                try:
                    after_mean_power=mean_power[file_idx+1,i]
                    count+=1
                except IndexError:
                    after_mean_power=0
                mean_power_neighbour=(prev_mean_power+after_mean_power)/count
                if current_mean_power>(1+self.tol)*mean_power_neighbour:
                    possible_flare[i]=True
            pos=np.where(possible_flare==True)[0]
            if len(pos)>=self.min_flare_freq:
                flare_detected[file_indx][pos]=True
                
        flare_list={}
        peak_flux=-1
        for file_idx in range(num_files):
            start_detected=np.where(np.isnan(low_pos_5sigma[file_idx,:])==False)[0]
            end_detected=np.where(np.isnan(high_pos_5sigma[file_idx,:])==False)[0]
            if len(start_detected)<self.min_flare_freq:
                flare_present=np.where(flare_detected[file_idx,:]==True)[0]
                if len(pos)<self.min_flare_freq:  #### flare is not detected even comparing
                                             #### neighbouring times
                    continue
                start_detected=np.where(np.isnan(low_pos_3sigma[file_idx,:])==False)[0]
                end_detected=np.where(np.isnan(high_pos_3sigma[file_idx,:])==False)[0]
                if len(start_detected)<self.min_flare_freq:
                    continue
                else:
                    peak_flux,peak_time,flare_start,flare_end=self.get_flare_params(flare_start_pos[file_idx,:],flare_end_pos[file_idx,:],\
                           peak_power[file_idx,:],peak_pos[file_idx,:])

            else:
                peak_flux,peak_time,flare_start,flare_end=self.get_flare_params(flare_start_pos[file_idx,:],flare_end_pos[file_idx,:],\
                        peak_power[file_idx,:],peak_pos[file_idx,:])
            if peak_flux>0:
                peak_time=Time(obs_times[file_idx][int(peak_time)],format='jd')
                flare_start=Time(obs_times[file_idx][int(flare_start)],format='jd')
                flare_end=Time(obs_times[file_idx][int(flare_end)],format='jd')
                obs_start=Time(obs_times[file_idx][0],format='jd')
                obs_end=Time(obs_times[file_idx][-1],format='jd')
                diff=flare_start-obs_start
                diff.format='sec'
                
                max_freq=np.argmax(peak_power[file_idx,:])
                if file_idx>0 and diff<60: ### if the flare start is extremely close to the start of the IDB file
                                          #### it is possible that the flare actually started earlier in the other
                                          #### IDB file. Since the files are sorted, I can just shift the index 
                                          #### by 1 and see if a lower threshold start is detected.
                    if np.isnan(low_pos_3sigma[file_idx-1,max_freq])==False:
                        #### Here I am trying to see if we can extend the flare start time to the previous IDB file.
                        probable_start=Time(obs_times[file_idx-1][int(low_pos_3sigma[file_idx-1,max_freq])],format='jd')
                        diff=flare_start-probable_start
                        diff.format='sec'
                        if diff<=self.max_diff_same_flare:
                            flare_start=probable_start
                diff=obs_end-flare_end
                diff.format='sec'

                if file_idx<num_files-1 and diff<10: #### doing similar thing for the flare end.
                    if np.isnan(low_pos_3sigma[file_idx+1,max_freq])==False: 
                        probable_end=Time(obs_times[file_idx+1][int(high_pos_3sigma[file_idx+1,max_freq])],format='jd')
                        diff=probable_end-flare_end
                        diff.format='sec'
                        if diff<=self.max_diff_same_flare:
                            flare_end=probable_end

                peak_time.format='isot'
                flare_start.format='isot'
                flare_end.format='isot'
                flare_list['flare_'+str(file_idx).zfill(2)]={'peak_flux':peak_flux,'flare_peak':peak_time,\
                        'flare_start':flare_start,'flare_end':flare_end}#,'freq_has_flare':flare_detected[file_idx,:]}
        print (flare_list)
        final_flares=self.combine_flares(flare_list)
        self.flare_list=final_flares
        self.flare_list_original=flare_list
        return

    def combine_flares(self,flare_list):
        '''
        This function tries to combine the detected flares based on their start time and end time and 
        produces a combined flare list.
        '''
        keys=list(flare_list.keys())
        flares={}
        removed_keys=[]
        for i,k in enumerate(keys):
            if k in removed_keys:
                continue
            current_start=flare_list[k]['flare_start']
            current_end=flare_list[k]['flare_end']
            current_peak_power=flare_list[k]['peak_flux']
            current_peak_time=flare_list[k]['flare_peak']
            for search_key in keys:
                if k==search_key or search_key in removed_keys:
                    continue
                diff1=flare_list[search_key]['flare_start']-current_start
                diff1.format='sec'
                diff2=flare_list[search_key]['flare_end']-current_start
                diff2.format='sec'
                #print (current_start,current_end,flare_list[search_key]['flare_start'],flare_list[search_key]['flare_end'])
                #print ("1",str(diff1.value)+","+str(diff2.value))
                if abs(diff1.value)<=self.max_diff_same_flare or \
                        abs(diff2.value)<=self.max_diff_same_flare:
                    if current_peak_power<flare_list[search_key]['peak_flux']:
                        flare_list[k]['peak_flux']=flare_list[search_key]['peak_flux']
                        flare_list[k]['flare_peak']=flare_list[search_key]['flare_peak']
                    if flare_list[search_key]['flare_start']<flare_list[k]['flare_start']:
                        flare_list[k]['flare_start']=flare_list[search_key]['flare_start']
                    
                    if flare_list[search_key]['flare_end']>flare_list[k]['flare_end']:
                        flare_list[k]['flare_end']=flare_list[search_key]['flare_end']

                    removed_keys.append(search_key)

                diff1=flare_list[search_key]['flare_end']-current_end
                diff1.format='sec'
                diff2=flare_list[search_key]['flare_start']-current_end
                diff2.format='sec'
                #print (current_start,current_end,flare_list[search_key]['flare_start'],flare_list[search_key]['flare_end'])
                #print ("2",str(diff1.value)+","+str(diff2.value))
                if abs(diff1.value)<=self.max_diff_same_flare or \
                        abs(diff2.value)<=self.max_diff_same_flare:
                    if current_peak_power<flare_list[search_key]['peak_flux']:
                        flare_list[k]['peak_flux']=flare_list[search_key]['peak_flux']
                        flare_list[k]['flare_peak']=flare_list[search_key]['flare_peak']
                    
                    if flare_list[search_key]['flare_start']<flare_list[k]['flare_start']:
                        flare_list[k]['flare_start']=flare_list[search_key]['flare_start']


                    if flare_list[search_key]['flare_end']>flare_list[k]['flare_end']:
                        flare_list[k]['flare_end']=flare_list[search_key]['flare_end']
                    removed_keys.append(search_key)


        for k in keys:
            if k in removed_keys:
                continue
            flares[k]=flare_list[k]
        return flares

    def get_flare_params(self,start_index,end_index,peak_power,peak_loc):
        '''
        I am assuming that there is only one flare within the timerange. Additionally I assume that all freqs
        peaks at similar times. If this assumption breaks down, there it might be possible that we divide the 
        entire time range into multiple chunks and search for flares there.

        I first calculate the median start and end for all the frequencies. Then I see foe how many frequencies
        the peak time at that frequency lie between the median start and end. If the number of such frequencies
        is greater than the min_flare_freq, I take the median as start and end time of the flare.
        '''
        start=np.nanmedian(start_index)
        end=np.nanmedian(end_index)
        count=0  
        for i in range(self.num_freqs):
            if peak_loc[i]>=start and peak_loc[i]<=end:
                count+=1
            else:
                peak_loc[i]=np.nan
        peak=np.nanmax(peak_power)
        if count>=self.min_flare_freq and peak>=self.min_flux_density:
            print ("peak="+str(peak))
            return peak,np.nanmedian(peak_loc),start,end
        return -1,-1,-1,-1


    def get_start_end_flare(self,peak_pos,norm_power,thresh,tot_times):
        '''
        I start from the peak position and proceed along both directions till
        I hit the threshold. These points give me the start and end of the flare.
        If I reach any edge before hitting the threshold, that boundary is said 
        to be the start/end.
        '''
        current_position=int(peak_pos-1)
        start=0
        end=0
        for i in range(current_position,-1,-1):
            try:
                if norm_power[i]<thresh:
                    start=i+1
                    break
            except IndexError:
                start=i+1
                break

        current_position=int(peak_pos+1)
        if peak_pos==tot_times-1:
            return start,peak_pos

        for i in range(current_position,tot_times):
            if norm_power[i]<thresh:
                end=i-1
                break
        if i==tot_times:
            end=tot_times-1
        return start,end
        
    def inspect(self,out,antlist):
        '''
            This is adapted from Dale's code in the EOVSA branch. This produces the list of
            good baselines and the data which is used for plotting. By good baselines Dale meant
            baselines whose length is between 150-1000 nsec.
        '''
        nt, = out['time'].shape
        blen = np.sqrt(out['uvw'][:,int(nt/2),0]**2 + out['uvw'][:,int(nt/2),1]**2)
        
        idx = []
        
        antlist.sort()
      
        for k,id1 in enumerate(antlist):
            for j in antlist[k+1:]:
                idx.append(ri.bl2ord[id1,j])
        idx = np.array(idx)
        good, = np.where(np.logical_and(blen[idx] > 150.,blen[idx] < 1000.))
        spec = np.nanmedian(np.abs(out['x'][idx[good],0]),0)
        return spec
        
    def make_plot(self):
        '''
            This code is adapted from Dale Gary's flare_spec.py code available in the EOVSA branch.
            This function does all the plotting related jobs. I makes the plot similar to the ones
            currently available in the EOVSA flare archive as of 2023/01/04. The upper panel shows 
            the dynamic spectrum and the lower panel shows the lightcurve at select frequencies. 
            The starttime of the plot is the flare start time-5 min
            The endtime of the plot is flare end + 5 min
            Dale in the comments of his code flare_spec.py wrote that the a 10s window is sufficient
            generally for dteremining the background. Here I use a 10s window. The 10 s window is 
            taken as 10 s before endtime and 10s after starttime (defined earlier). I choose the
            minimum of these two quantities as the background.
            For plotting, I set vmax and vmin as the 10 and 98 percentiles of the dynamic spectrum.
            The frequency axis is in log scale. 
            The colorscale is also in log.
            In the lower panel, the yaxis is auto-scaled and is in linear scale.
            All flux are in SFU.

        '''
        from astropy.time import TimeDelta
        from matplotlib.dates import DateFormatter
        from astropy.visualization import LogStretch, ImageNormalize
        nf=np.size(self.freqs)
        #nt=np.size(self.times)
        diff=TimeDelta(self.time_diff,format='sec')

        for flare_key in self.flare_list.keys():
            flare=self.flare_list[flare_key]
            starttime=flare['flare_start']-diff
            endtime=flare['flare_end']+diff
            #### detecting the start and end index required for plotting. 
            for j,time1 in enumerate(self.times):
                if time1>=starttime:
                    start_id=j
                    break
            for j,time1 in enumerate(self.times[start_id+1:]):
                if time1>=endtime:
                    end_id=j+start_id
                    break
            spec=self.spec_for_plotting[:,start_id:end_id+1]
            nt=spec.shape[1] 
            bgd1=None
            bgd2=None
            #### Determining the background
            try:
                bgidx1=[0,10]
                bgd1 = np.nanmean(spec[:,bgidx1[0]:bgidx1[1]],1)
            except:
                pass
            try:
                bgidx2=[-11,-1]
                bgd2 = np.nanmean(spec[:,bgidx2[0]:bgidx2[1]],1)
            except:
                pass
        
            if bgd1.any()==None and bgd2.any()==None:
                from copy import deepcopy
                subspec=deepcopy(spec)
            elif bgd1.any()==None:
                subspec=spec-bgd2.repeat(nt).reshape(nf,nt)
            elif bgd2.any()==None:
                subspec=spec-bgd1.repeat(nt).reshape(nf,nt)
            else:
                avg_bgd1=np.nanmedian(bgd1)
                avg_bgd2=np.nanmedian(bgd2)
                bgd=bgd1
                if avg_bgd2<avg_bgd1:
                    bgd=bgd2
                subspec=spec-bgd.repeat(nt).reshape(nf,nt)  #### Backgroudn subtraction done
         
            # Next two lines force a gap in the plot for the notched frequencies (does nothing for pre-2019 data)
            bad, = np.where(abs(self.freqs - 1.742) < 0.001)
            if len(bad) > 0: subspec[bad] = np.nan


            def fix_times(jd):
                bad, = np.where(np.round((jd[1:] - jd[:-1])*86400) < 1)
                for b in bad:
                    jd[b+1] = (jd[b] + jd[b+2])/2.
                return jd  #### TODO changed here. Probable issue in Dale's code. Confirm
        
            times1=self.times[start_id:end_id+1]

            jd=times1.value
            jd=fix_times(jd)
            times = Time(jd,scale='utc',format='jd')
            # Make sure time gaps look like gaps
            gaps = np.where(np.round((jd[1:] - jd[:-1])*86400) > 1)
            for gap in gaps:
                subspec[:,gap] = np.nan
        
            f = plt.figure(figsize=[14,8])
            ax0 = plt.subplot(211)
            ax1 = plt.subplot(212)
            max_spec=np.nanmax(subspec)
            min_spec=np.nanmin(subspec)
        
            ### Overriding the user inputs in case they are higher/lower than the DS max/min
            plot_limits=np.nanpercentile(subspec,[10,99])
            norm=ImageNormalize(vmax=plot_limits[1],vmin=plot_limits[0],stretch=LogStretch(10))
            im2 = ax0.pcolormesh(times.plot_date,self.freqs,subspec,norm=norm)
            for frq in self.lcfreqs:
                lc = np.nanmean(subspec[frq-5:frq+5],0)
                ax1.step(times.plot_date,lc,label=str(self.freqs[frq])[:6]+' GHz')
            #ax1.set_ylim(-0.5,self.vmax)
            ax1.xaxis_date()
            ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
            ax0.xaxis_date()
            ax1.set_xlabel('Time [UT]')
            ax1.set_ylabel('Flux Density [sfu]')
            ax0.set_ylabel('Frequency [GHz]')
            ax0.set_title('EOVSA Data for '+times[0].iso[:10])
            ax0.xaxis.set_major_formatter(DateFormatter("%H:%M"))
            ax1.set_xlim(times[[0,-1]].plot_date)
            ax0.set_xlim(times[[0,-1]].plot_date)
            ax1.legend()
            ax0.set_yscale('log')
            ax0.set_ylim([self.freqs[0],self.freqs[-1]])
            if self.name is None:
                figname=self.outpath+flare_key  #### TODO decide a naming convetion. Moving ahead with a simple name.
                                   #### Not suitable for production runs.
            else:
                figname=self.outpath+self.name+"_"+flare_key
            f.savefig(figname+'.png')
            fh = open(figname+'.dat','wb')
            fh.write(times.value)
            fh.write(self.freqs)
            fh.write(subspec)
            fh.close()
        return
