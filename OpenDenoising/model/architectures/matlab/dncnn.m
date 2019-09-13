% Copyright or © or Copr. IETR/INSA Rennes (2019)
%
% Contributors :
%     Eduardo Fernandes-Montesuma eduardo.fernandes-montesuma@insa-rennes.fr (2019)
%     Florian Lemarchand florian.lemarchand@insa-rennes.fr (2019)
%
%
% OpenDenoising is a computer program whose purpose is to benchmark image
% restoration algorithms.
%
% This software is governed by the CeCILL-C license under French law and
% abiding by the rules of distribution of free software. You can  use,
% modify and/ or redistribute the software under the terms of the CeCILL-C
% license as circulated by CEA, CNRS and INRIA at the following URL
% "http://www.cecill.info".
%
% As a counterpart to the access to the source code and rights to copy,
% modify and redistribute granted by the license, users are provided only
% with a limited warranty  and the software's author, the holder of the
% economic rights, and the successive licensors have only  limited
% liability.
%
% In this respect, the user's attention is drawn to the risks associated
% with loading, using, modifying and/or developing or reproducing the
% software by the user in light of its specific status of free software,
% that may mean  that it is complicated to manipulate,  and  that  also
% therefore means  that it is reserved for developers  and  experienced
% professionals having in-depth computer knowledge. Users are therefore
% encouraged to load and test the software's suitability as regards their
% requirements in conditions enabling the security of their systems and/or
% data to be ensured and, more generally, to use and operate it in the
% same conditions as regards security.
%
% The fact that you are presently reading this means that you have had
% knowledge of the CeCILL-C license and that you accept its terms.



function [layers] = dncnn(varargin)
% DNCNN creates computational graph for DnCNN network using Matlab's Deep Learning toolbox
%   net = dncnn() creates DnCNN default implementation, with 17 layers, input shape [40, 40, 1],
%                 3 x 3 filters and 64 filters per convolutional layer.
%
%   Options:
%             'Depth'                   - Number of layers in DnCNN. Default 17.
%             'sizePatch'               - Input shape will be [sizePatch, sizePatch, numChannels]. Default 40.
%             'sizeFilters'             - Size of filters employed in convolution. Default 3.
%             'numFilters'              - Number of filters in each convolutional layer. Default 64.
%             'numChannels'             - Number of input channels. Default 1.

%% Parsing arguments
parser = inputParser;
addOptional(parser, 'Depth', 17);
addOptional(parser, 'sizePatch', 40);
addOptional(parser, 'sizeFilters', 3);
addOptional(parser, 'numFilters', 64);
addOptional(parser, 'numChannels', 1);
parse(parser, varargin{:});

%% Creating Neural Network
inputSize = [parser.Results.sizePatch, parser.Results.sizePatch, parser.Results.numChannels];

layers = [
    imageInputLayer(inputSize, 'name', 'input_img', 'Normalization', 'none')
    convolution2dLayer(parser.Results.sizeFilters, parser.Results.numFilters, 'name', 'conv_1', ...
                       'Padding', 'same', 'BiasL2Factor', 0)
    reluLayer('name', 'relu_1')
];

layers(2).Weights = sqrt(2 / (parser.Results.sizeFilters * parser.Results.sizeFilters * parser.Results.numFilters)) * ...
                    randn(parser.Results.sizeFilters, parser.Results.sizeFilters, parser.Results.numChannels, ...
                          parser.Results.numFilters);
layers(2).Bias = zeros(1, 1, parser.Results.numFilters);

for i=2:parser.Results.Depth - 1
    layers = [
        layers
        convolution2dLayer(parser.Results.sizeFilters, parser.Results.numFilters, 'name',...
                           strcat('conv_', num2str(i)), ...
                           'BiasLearnRateFactor', 0, ...
                           'BiasL2Factor', 0, ...
                           'Stride', [1, 1], ...
                           'Padding', 'same')
        batchNormalizationLayer('name', strcat('batchnorm_', num2str(i)), ...
                                'Offset',zeros(1, 1, parser.Results.numFilters),...
                                'OffsetL2Factor',0, ...
                                'ScaleL2Factor', 0)
        reluLayer('name', strcat('relu_', num2str(i)))
    ];

    j = 3 * (i - 1) + 1;
    layers(j).Weights = sqrt(2 / (parser.Results.sizeFilters * parser.Results.sizeFilters * parser.Results.numFilters)) * ...
                        randn(parser.Results.sizeFilters, parser.Results.sizeFilters, parser.Results.numFilters, ...
                              parser.Results.numFilters);
    layers(j).Bias = zeros(1, 1, parser.Results.numFilters);


end

layers = [
    layers
    convolution2dLayer(parser.Results.sizeFilters, 1, ...
                       'name', strcat('conv_', num2str(parser.Results.Depth)), ...
                       'Stride', [1, 1], ...
                       'Padding', 'same')
    regressionLayer('Name','routput')
];

layers(end - 1).Weights = sqrt(2 / (parser.Results.sizeFilters * parser.Results.sizeFilters * parser.Results.numFilters)) * ...
                          randn(parser.Results.sizeFilters, parser.Results.sizeFilters, parser.Results.numFilters, ...
                                parser.Results.numChannels);
layers(end - 1).Bias = zeros(1, 1, parser.Results.numChannels);

end
