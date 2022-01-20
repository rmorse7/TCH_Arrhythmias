clear all
clc
% [filename,pathname] = uigetfile('*.*','Select the ECG Signal');
% filewithpath = strcat(pathname,filename);
% Fs = input('Enter Sampling Rate: ');
% 
% ecg = load('I:\COMP549\mit_bih_database\100m.mat'); % Reading ECG signal
% ecgsig = (ecg.val)./200; % Normalize gain for mit_bih database the gain is 200
%%%%%%%%%%%%%%%
colormap = jet(128);
signallength = 600;
fs = 240;
fb = cwtfilterbank('SignalLength',signallength,'Wavelet','amor','VoicesPerOctave',12);
folder = 'C:\YiweiZhu\ricephd\course\COMP 549\data\1\';
filename = 'Reference_idx_1_Time_block_4.h5';

info = h5info([folder filename]);
h5disp(info.Filename);

ECG2 = h5read(info.Filename,[info.Name 'GE_WAVE_ECG_2_ID']);
time = h5read(info.Filename,[info.Name 'time']);
PJ_start_idx = 1449229;



ecg = double(ECG2(1:600));
ecg_jet = double(ECG2(PJ_start_idx:PJ_start_idx+599));
cfs = abs(fb.wt(ecg));
sub_time = linspace(1,600,600);
im = ind2rgb(im2uint8(rescale(cfs)),colormap);
figure(1)
subplot(2,1,1);
plot(sub_time,ecg)
xlabel('Data points')
ylabel('ECG(mV) ')
subplot(2,1,2); 

imagesc(im)
title('Continuous Wavelet Coefficients Scalogram')

cfs_pj = abs(fb.wt(ecg_jet));
im_pj = ind2rgb(im2uint8(rescale(cfs_pj)),colormap);
figure(2)
subplot(2,1,1);
plot(sub_time,ecg_jet)
xlabel('Data points')
ylabel('ECG(mV) ')
subplot(2,1,2);
imagesc(im_pj);
title('Continuous Wavelet Coefficients Scalogram')








%filename = 'p1e1_test.jpg';
%imwrite(imresize(im,[227,227]),filename);
%Fs = 240;