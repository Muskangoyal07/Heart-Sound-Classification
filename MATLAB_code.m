datasetPath = 'C:\Users\itsmu\Desktop\NITR\Heart sound classification\PCG\PCG_Data\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0';
folders = {'training-a', 'training-b', 'training-c', 'training-d', 'training-e'};
normalData = {};
abnormalData = {};
for folderIdx = 1:length(folders)
    folderName = folders{folderIdx};
    labelFile = fullfile(datasetPath, folderName, 'REFERENCE.csv');
    labelTable = readtable(labelFile, 'ReadVariableNames', false);
    for fileIdx = 1:height(labelTable)
        filename = fullfile(datasetPath, folderName, [labelTable.Var1{fileIdx}, '.wav']);
        [signal, fs] = audioread(filename);
        label = labelTable.Var2(fileIdx);
        if label == 1
            abnormalData{end+1} = signal;
        else
            normalData{end+1} = signal;
        end
    end
end
numNormal = numel(normalData);
numAbnormal = numel(abnormalData);
fprintf('Number of normal data: %d\n', numNormal);
fprintf('Number of abnormal data: %d\n', numAbnormal);
mkdir('pcgdataset11');
mkdir('pcgdataset11/normal');
mkdir('pcgdataset11/abnormal');
[b, a] = butter(6, [25 600] / (fs / 2), 'bandpass');
numBins = 59;
function compute_and_plot_lbp(signal, fs, filename, b, a, numBins)
    signal = double(signal);
    signal = filtfilt(b, a, signal);
    signal = (signal - mean(signal)) / std(signal);
    signal = wdenoise(signal, 'Wavelet', 'coif5', 'DenoisingMethod', 'UniversalThreshold');
    window = hann(256);
    noverlap = 128;
    nfft = 512;
    [S, F, T] = spectrogram(signal, window, noverlap, nfft, fs);
    S_magnitude = abs(S);
    S_gray = mat2gray(10 * log10(S_magnitude));
    lbp_features = extractLBPFeatures(S_gray, 'Upright', false);
    fig = figure('Visible', 'off');
    ax1 = subplot(2,1,1);
    imagesc(T, F, 10*log10(S_magnitude));
    set(gca, 'YDir', 'normal');
    colormap(jet);
    axis off;
    set(gca, 'Position', [0, 0.5, 1, 0.5]); 
    ax2 = subplot(2,1,2);
    histogram(lbp_features, numBins);
    axis off;
    set(gca, 'Position', [0, 0, 1, 0.5]);
    set(gcf, 'Units', 'pixels', 'Position', [0, 0, 145, 145]);
    tightfig();
    saveas(fig, filename);
    close(fig);
end
for i = 1:numel(normalData)
    signal = normalData{i};
    filename = fullfile('pcgdataset11/normal', ['n' num2str(i) '.png']);
    compute_and_plot_lbp(signal, fs, filename, b, a, numBins);
end
for i = 1:numel(abnormalData)
    signal = abnormalData{i};
    filename = fullfile('pcgdataset11/abnormal', ['ab' num2str(i) '.png']);
    compute_and_plot_lbp(signal, fs, filename, b, a, numBins);
end
function tightfig()
    fig = gcf;
    ax = findall(fig, 'Type', 'axes');
    for i = 1:length(ax)
        outerpos = ax(i).OuterPosition;
        ti = ax(i).TightInset;
        left = outerpos(1) + ti(1);
        bottom = outerpos(2) + ti(2);
        ax_width = outerpos(3) - ti(1) - ti(3);
        ax_height = outerpos(4) - ti(2) - ti(4);
        ax(i).Position = [left, bottom, ax_width, ax_height];
    end
end
DatasetPath = 'C:\Users\itsmu\Desktop\NITR\Heart sound classification\PCG\PCG_Data\pcgdataset11';
images = imageDatastore(DatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[TrainImages, TestImages] = splitEachLabel(images, 0.8);  
net = alexnet;
layersTransfer = net.Layers(1:end-3);
numClasses = 2;
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...             
    'MaxEpochs', 20,...                 
    'InitialLearnRate', 1e-4, ...        
    'Shuffle', 'every-epoch', ...        
    'ValidationData', TestImages, ...   
    'ValidationFrequency', 100, ...        
    'Verbose', false, ...                 
    'Plots', 'training-progress');       
netTransfer = trainNetwork(TrainImages, layers, options);
YPred = classify(netTransfer, TestImages);
YValidation = TestImages.Labels;
figure;
plotconfusion(YValidation, YPred);
C = confusionmat(YValidation, YPred);
TP = C(1,1);
FN = C(1,2);
FP = C(2,1);
TN = C(2,2);
accuracy_percentage = (sum(YPred == YValidation) / numel(YValidation)) * 100;
sensitivity = TP / (TP + FN);
specificity = TN / (TN + FP);
precision = TP / (TP + FP);
F1Score = 2 * (precision * sensitivity) / (precision + sensitivity);
sensitivity_percentage = sensitivity * 100;
specificity_percentage = specificity * 100;
precision_percentage = precision * 100;
F1Score_percentage = F1Score * 100;
disp(['Accuracy: ', num2str(accuracy_percentage), '%']);
disp(['Sensitivity: ', num2str(sensitivity_percentage), '%']);
disp(['Specificity: ', num2str(specificity_percentage), '%']);
disp(['Precision: ', num2str(precision_percentage), '%']);
disp(['F1 Score: ', num2str(F1Score_percentage), '%']);

