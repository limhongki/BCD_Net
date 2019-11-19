def train_matlab(layer_idx, kern_size=3, kern_num=81, pad_size=1, lr_enc=1e-4, lr_dec=1e-4, lr_threshold=1e-4, alpha_init=-15, num_epoch=5,
          Lbatch=1, train_data_name='sphere', test_data_name='liver', step_size=400, gamma=0.1):

    import os as os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parameter import Parameter
    import numpy as np
    from torch.autograd import Variable
    from torch.utils.data import Dataset as dset
    from torch.utils.data import DataLoader
    from scipy.io import loadmat
    import scipy
    
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.chdir('/n/escanaba/v/hongki/Documents/BCD-Net/')
    print('cwd is:' + os.getcwd())

    class mysoftshrink(nn.Module):
        def __init__(self, K, initial_threshold=-15):
            super(mysoftshrink, self).__init__()
            self.K = K
            self.initial_threshold = initial_threshold
            self.alpha = Parameter(initial_threshold * torch.ones(1, K, 1, 1, 1))

        def forward(self, input):
            return (input.abs() > torch.exp(self.alpha)).type(dtype) * (
                        input.abs() - torch.exp(self.alpha)) * input.sign()

    class autoEncoder(nn.Module):
        def __init__(self):
            super(autoEncoder, self).__init__()
            self.encoder = nn.Conv3d(1, kern_num, kern_size, padding=pad_size, bias=False)
            self.soft_thresholding = mysoftshrink(kern_num, alpha_init)
            self.decoder = nn.Conv3d(kern_num, 1, kern_size, padding=pad_size, bias=False)

        def forward(self, x):
            x = self.encoder(x)
            x = self.soft_thresholding(x)
            u = self.decoder(x)
            return u

    class mydataset(dset):
        def __init__(self, folderpath_img, test):
            super(mydataset, self).__init__()
            if test == True:
                append = '_test'
            else:
                append = ''
            self.I_true_all = loadmat(folderpath_img)["Itrue" + append].transpose(3, 2, 1, 0).astype(float)
            self.I_noisy_all = loadmat(folderpath_img)["Irecon" + append].transpose(3, 2, 1, 0).astype(float)

        def __len__(self):
            return len(self.I_true_all)  # number of samples

        def __getitem__(self, index):
            I_true = np.expand_dims(self.I_true_all[index], axis=0)
            I_noisy = np.expand_dims(self.I_noisy_all[index], axis=0)
            return (I_true, I_noisy)

    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    train_dataset = mydataset('mypcodes/cache/Training_data_' + train_data_name + '_Layer' + str(layer_idx) + '.mat', test=False)
    train_loader = DataLoader(train_dataset, batch_size=Lbatch, shuffle=False)

    test_dataset = mydataset('mypcodes/cache/Testing_data_' + test_data_name + '_Layer' + str(layer_idx) + '.mat', test=True)
    test_loader = DataLoader(test_dataset, batch_size=Lbatch, shuffle=False)

    torch.manual_seed(100)

    net = autoEncoder()
    print(get_n_params(net))
    dtype = torch.cuda.FloatTensor
    net.type(dtype)
    net = nn.DataParallel(net)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam([
        {'params': net.module.encoder.parameters(), 'lr': lr_enc},
        {'params': net.module.decoder.parameters(), 'lr': lr_dec},
        {'params': net.module.soft_thresholding.parameters(), 'lr': lr_threshold}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_history = []
    test_loss_history = []
    update_ratio_W_hitory = []
    update_ratio_alpha_hitory = []
    update_ratio_D_hitory = []

    device = 'cuda'
    old_W = torch.cuda.FloatTensor(net.module.encoder.weight.shape)
    old_alpha = torch.cuda.FloatTensor(net.module.soft_thresholding.alpha.shape)
    old_D = torch.cuda.FloatTensor(net.module.decoder.weight.shape)
    
    for epoch in range(num_epoch):
        losses = []
        net.train()
        for idx, data in enumerate(train_loader, 0):
            I_true_bat, I_noisy_bat = data
            I_true_bat = I_true_bat.to(device).float()
            I_noisy_bat = I_noisy_bat.to(device).float()
            IM_denoised = net(I_noisy_bat)
            
            old_W.copy_(net.module.encoder.weight.data)
            old_alpha.copy_(net.module.soft_thresholding.alpha.data)
            old_D.copy_(net.module.decoder.weight.data)
            loss = criterion(IM_denoised, I_true_bat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            update_ratio_W = torch.norm(old_W - net.module.encoder.weight.data) / torch.norm(old_W)
            update_ratio_D = torch.norm(old_D - net.module.decoder.weight.data) / torch.norm(old_D)
            update_ratio_alpha = torch.norm(old_alpha - net.module.soft_thresholding.alpha.data) / torch.norm(old_alpha)
            update_ratio_W_hitory.append(update_ratio_W.item())
            update_ratio_alpha_hitory.append(update_ratio_alpha.item())
            update_ratio_D_hitory.append(update_ratio_D.item())

        scheduler.step()
        alpha_epoch = net.module.soft_thresholding.alpha.data.cpu().numpy()
        print('Current epoch: %d || Loss: %E || Update ratio W:%3E, alpha:%3E, D:%3E, alpha max: %g , alpha min: %g' % (
            (epoch + 1), np.mean(losses), np.mean(update_ratio_W_hitory), np.mean(update_ratio_alpha_hitory),
            np.mean(update_ratio_D_hitory), np.amax(alpha_epoch), np.amin(alpha_epoch)))

    net.eval()

    for idx, data in enumerate(train_loader, 0):
        I_true_bat, I_noisy_bat = data
        I_true_bat = I_true_bat.to(device).float()
        I_noisy_bat = I_noisy_bat.to(device).float()
        IMout = net(I_noisy_bat)
        IMout = IMout.permute(4,3,2,1,0)
        IMout = IMout.data.cpu().numpy()
        scipy.io.savemat(
            'mypcodes/cache/IMout_' + train_data_name + '_Layer' + str(layer_idx + 1) + '_image_' + str(idx) + '.mat',
            mdict={'IMout': IMout})

    for idx, data in enumerate(test_loader, 0):
        I_true_test_bat, I_noisy_test_bat = data
        I_true_test_bat = I_true_test_bat.to(device).float()
        I_noisy_test_bat = I_noisy_test_bat.to(device).float()
        IMout_test = net(I_noisy_test_bat)
        IMout_test = IMout_test.permute(4,3,2,1,0)
        IMout_test = IMout_test.data.cpu().numpy()
        scipy.io.savemat(
            'mypcodes/cache/IMout_test_' + test_data_name + '_Layer' + str(layer_idx + 1) + '_image_' + str(idx) + '.mat',
            mdict={'IMout_test': IMout_test})
    scipy.io.savemat('mypcodes/cache/model_weights_layer' + str(layer_idx + 1) + '.mat', mdict={'W':net.module.encoder.weight.data.cpu().numpy(), 'D':net.module.decoder.weight.data.cpu().numpy(), 'alpha':net.module.soft_thresholding.alpha.data.cpu().numpy()})





