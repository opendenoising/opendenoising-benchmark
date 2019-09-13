function [image_restored] = dncnn_denoise(net_path, image, useGPU)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Given an image array, simplenn_denoise performs image denoising by     %
% loading a deep neural network architecture located on 'net_path'.      %
% Params:                                                                %
%   1) net_path: path to .m file containing the model.                   %
%   2) image: 2d float array containing the image                        %
%   3) useGPU: boolean defining if it will use GPU or not.               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load Network architecture
net_obj = load(net_path);
net = net_obj.net;
net = vl_simplenn_tidy(net);

%% Move net to GPU
if useGPU == 1
    net = vl_simplenn_move(net, 'gpu'); % Pass net to GPU
    image = gpuArray(image);            % Pass image to GPU
end

%% Perform denoising
res = vl_simplenn(net, image, [], [], 'conserveMemory', true, 'mode', 'test');
image_restored = res(end).x;

%% Move result to CPU
if useGPU == 1
	image = gather(image);
    image_restored = gather(image_restored);
end

image_restored = single(image - image_restored);

end