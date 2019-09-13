function [image_restored] = MWCNN_denoise(net_path, image, useGPU)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Given an image array, mwcnn performs image denoising by loading        %
% a deep neural network architecture located on 'net_path'.              %
% Params:                                                                %
%   1) net_path: path to .m file containing the model.                   %
%   2) image: 2d float array containing the image                        %
%   3) useGPU: boolean defining if it will use GPU or not.               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load Network architecture
net_obj = load(net_path);
net = net_obj.net;
net = dagnn.DagNN.loadobj(net);
net.removeLayer('objective');
out_idx = net.getVarIndex('prediction');
net.vars(net.getVarIndex('prediction')).precious = 1;
net.mode = 'test';

%% Preprocess image
input = image * 4 - 2;

%% Move net to GPU
if useGPU == 1
    net.move('gpu');    % Pass net to GPU
    input = gpuArray(input);  % Pass image to GPU
end

%% Computes output
net.eval({'input', input});
image_restored = gather(...
    squeeze(...
        gather(...
            net.vars(out_idx).value + 2 ...
        ) / 4 ...
    )...
);

end

