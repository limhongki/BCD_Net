clear all; close all; clc;

%% Construct paired training data
pcodes_init

cache_path = 'mypcodes/cache/';
mkdir('./result')
mkdir('./recon')
load testing_data.mat;
load training_data.mat

train_data_name = 'xcat_train';
test_data_name = 'xcat_test';
niter = 1;
num_out_iter = 30;     
Irecon_ls = Irecon_orig(:,:,:,end-niter+1:end,:);
Irecon_ls_test = Irecon_orig_test(:,:,:,end-niter+1:end,:);

filename = strcat(cache_path,['Training_data_',train_data_name,'_Layer0.mat']);
save(filename,'Irecon','Itrue', '-v7')

filename = strcat(cache_path,['Testing_data_',test_data_name,'_Layer0.mat']);
save(filename,'Irecon_test','Itrue_test', '-v7')

gen_sys;

ld = 2^2;
R = 3^2;     
K = 78;    

lr_enc = 1e-2;
lr_dec = 1e-3;
lr_threshold = 1e-1;  
step_size = 400;
gamma = 0.1;
num_epoch = 500;   %number of iterations of alternating minimization for training 1 layer

Rpad = floor((sqrt(R)-1)/2);    %padding size
Lbatch = 1;       %size of mini batch
%% Learn image mapping autoencoder and apply reconstruction modules
L = size(Itrue,4);
Ltest = size(Itrue_test,4);
RMSE = zeros(num_out_iter,L,'single');                    %RMSE for training
RMSEtest = zeros(num_out_iter,Ltest,'single');

%Learn autoencoder for "num" different recon layers
disp(['Training begins. (L=', num2str(L), ', Lbatch=', num2str(Lbatch), ')']);

for kp = 1:num_out_iter
    xx = mean(Irecon,4);
    x_sorted = sort(xx(:),'descend');
    alpha_init = x_sorted(round(length(x_sorted)*10/100));
    alpha_init = log(alpha_init);
    
    py.mypcodes.train_matlab.train_matlab(int32(kp-1),...
        pyargs('kern_size',int32(sqrt(R)), 'kern_num',int32(K), 'pad_size',int32(Rpad),...
        'lr_enc',lr_enc, 'lr_dec',lr_dec, 'lr_threshold',lr_threshold,...
        'alpha_init', alpha_init, 'num_epoch',int32(num_epoch), 'Lbatch',int32(Lbatch), 'train_data_name', train_data_name, 'test_data_name', test_data_name, 'step_size', step_size, 'gamma', gamma));
    
    for i = 1:size(Irecon,4)        
        ci = ci_ls(:,:,:,i);
        Gb = Gblock(G, 1, 'odiag', ci);
        filename = ['IMout_',train_data_name,'_Layer',num2str(kp),'_image_',num2str(i-1),'.mat'];
        load(['./mypcodes/cache/',filename])
        xhat_ls = func_minimize_fx_rx(Irecon(:,:,:,i), Gb, yi_ls(:,:,:,i), ri_ls(:,:,:,i), IMout, ld, niter);
        Irecon(:,:,:,i) = xhat_ls(:,:,:,end);
        Irecon_next_iter(:,:,:,:,i) = xhat_ls;
    end
    Irecon_ls = cat(4,Irecon_ls,Irecon_next_iter);
    
    for i = 1:size(Irecon_test,4)        
        ci = ci_ls_test(:,:,:,i);
        Gb = Gblock(G, 1, 'odiag', ci);
        filename = ['IMout_test_',test_data_name,'_Layer',num2str(kp),'_image_',num2str(i-1),'.mat'];
        load(['./mypcodes/cache/',filename])
        xhat_ls = func_minimize_fx_rx(Irecon_test(:,:,:,i), Gb, yi_ls_test(:,:,:,i), ri_ls_test(:,:,:,i), IMout_test, ld, niter);
        Irecon_test(:,:,:,i) = xhat_ls(:,:,:,end);
        Irecon_next_iter_test(:,:,:,:,i) = xhat_ls;        
    end
    Irecon_ls_test = cat(4,Irecon_ls_test,Irecon_next_iter_test);
    
    %Record performances    
    RMSE(kp,:) = reshape(cell2mat( cellfun( @(A,B)(my_rmse(A,B,ig)), ...
        num2cell(Irecon, [1 2 3]), num2cell(Itrue, [1 2 3]), 'UniformOutput', false') ),[1,L]);
    RMSEtest(kp,:) = reshape(cell2mat( cellfun( @(A,B)(my_rmse(A,B,ig)), ...
        num2cell(Irecon_test, [1 2 3]), num2cell(Itrue_test, [1 2 3]), 'UniformOutput', false') ),[1,Ltest]);
    subplot(121);im(Irecon(:,:,:,1));title(sprintf('Layer: %g, Irecon, RMSE: %g', kp, mean(RMSE(kp,:))));drawnow;
    subplot(122);im(Irecon_test(:,:,:,1));title(sprintf('Layer: %g, Irecon, RMSE: %g', kp, mean(RMSEtest(kp,:))));drawnow;
    
    %Save the training and testing datasets to train the autoencoder in the next layer
    filename = strcat(cache_path,'Training_data_',train_data_name,'_Layer',num2str(kp),'.mat');
    save(filename,'Irecon','Itrue','-v7');    
    filename = strcat(cache_path,'Testing_data_',test_data_name,'_Layer',num2str(kp),'.mat');
    save(filename,'Irecon_test','Itrue_test','-v7');     
    disp(['BCD Layer #', num2str(kp),' RMSE:', num2str(mean(RMSEtest(kp,:)))]);
    name = ['sca_R_', num2str(R), '_K_',num2str(K), '_ld_', num2str(ld), '_layer_', num2str(num_out_iter), '_xiter_', num2str(niter), '_epoch_',num2str(num_epoch),'_lr_',num2str(lr_dec),'_',num2str(lr_enc),'_',num2str(lr_threshold),'.mat'];
    fld_write(['./recon/xbcd_net_',name,'.fld'], Irecon_ls_test)
end

clear yi_ls ri_ls ci_ls yi_ls_test ri_ls_test ci_ls_test Irecon_next_iter Irecon_next_iter_test
save(['./result/',name], '-v7.3')
