clear all
clc
% Function to convert masked images exported from QuPath to instances

% function qupath_to_mask(varargin)
% nucleus_train_val_split = 0.85;
% unet_train_val_split = 0.85;
nuc_output_size = 1024;         % Set this to < 2 to output nuclei images of different sizes for maskrcnn
imgpath = './human-pathology/ndpi-files/02_masks';
nucleus_out_path = './human-pathology/ndpi-files/03_nucleus/';
tissue_out_path = './human-pathology/ndpi-files/04_tissue/';
tissue_stage1_out_path = './human-pathology/ndpi-files/05_tissue_stage1/';
tissue_unet_out_path = './human-pathology/ndpi-files/06_tissue_unet/';
filenames = dir(imgpath); filenames = filenames(3:end);
class_id = {'Nucleus', 'Tubules', 'Glomerulus', 'Vessel'};
nuclei_trn_img = {};
nuclei_mask = {};
tissue_trn_img = {};
tissue_mask = {};

% List of files to output: [nuclei_images, tissue_images, tissue_images_stage1, unet_images]
output_list = [0, 0, 0, 1];


for i = 1:numel(filenames)
    name = filenames(i).name;
    tmp = regexp(name, '_(\w{3,12})_\(1.00,(\w{1,6}),(\w{1,6}),(\w{1,6}),(\w{1,6})\)-{0,1}(\w{0,6})','tokens');
    try
        tmp = tmp{1};
    catch
        disp(name)
    end
    tmp{7} = name;

    % Nucleus Train Images
    if strcmp(tmp{1}, 'NucleusTrain') && ~strcmp(tmp{6}, 'mask')
        nuclei_trn_img{end+1} = tmp;
    elseif strcmp(tmp{1}, 'Nucleus') && strcmp(tmp{6}, 'mask')
        nuclei_mask{end+1} = tmp;
    end

    % Other Tissue Train Images
    if strcmp(tmp{1}, 'TissueTrain') && ~strcmp(tmp{6}, 'mask')
        tissue_trn_img{end+1} = tmp;
    elseif ismember({tmp{1}}, {class_id{2:end}}) && strcmp(tmp{6}, 'mask')
        tissue_mask{end+1} = tmp;
    end
end
    
%% Load nuclei training image and create instance labels
if output_list(1)
    for i = 1:numel(nuclei_trn_img)
        img = imread([imgpath '/' nuclei_trn_img{i}{7}]);
        
        if min(size(img,1),size(img,2)) >= nuc_output_size
            [pathstr, name, ext] = fileparts(nuclei_trn_img{i}{7});
            mkdir([nucleus_out_path, name, '/images/']);
            imwrite(img(1:nuc_output_size,1:nuc_output_size, :), [nucleus_out_path, name, '/images/', name, '.png']);

            jj = 0;
            for j = 1:numel(nuclei_mask)
                img_prefix = strsplit(nuclei_trn_img{i}{7}, '_'); img_prefix = img_prefix{1};
                mask_prefix = strsplit(nuclei_mask{j}{7}, '_'); mask_prefix = mask_prefix{1};

                if strcmp(img_prefix, mask_prefix)
                    % Now assign Nuclei instances to this image
                    x_start = str2double(nuclei_mask{j}{2}) - str2double(nuclei_trn_img{i}{2}) + 1;
                    y_start = str2double(nuclei_mask{j}{3}) - str2double(nuclei_trn_img{i}{3}) + 1;
                    x_end = str2double(nuclei_trn_img{i}{2}) + min(str2double(nuclei_trn_img{i}{4}),nuc_output_size) - str2double(nuclei_mask{j}{2}) - str2double(nuclei_mask{j}{4});
                    y_end = str2double(nuclei_trn_img{i}{3}) + min(str2double(nuclei_trn_img{i}{5}),nuc_output_size) - str2double(nuclei_mask{j}{3}) - str2double(nuclei_mask{j}{5});

                    if (x_start > 0 && y_start > 0 && x_end > 0 && y_end > 0)
                        instances = uint8(zeros(size(img(1:nuc_output_size,1:nuc_output_size, :))));
                        mask = imread([imgpath '/' nuclei_mask{j}{7}]);
                        instances(y_start:size(instances,1)-y_end, x_start:size(instances,2)-x_end) = mask(:,:,1);
                        mkdir([nucleus_out_path, name, '/masks_001/']);
                        imwrite(instances(1:nuc_output_size,1:nuc_output_size), [nucleus_out_path, name, '/masks_001/', name, '_', sprintf('%04d', jj), '.png']);
                        jj = jj+1;
                    end
                end
            end
        elseif nuc_output_size < 2
            [pathstr, name, ext] = fileparts(nuclei_trn_img{i}{7});
            mkdir([nucleus_out_path, name, '/images/']);
            imwrite(img, [nucleus_out_path, name, '/images/', name, '.png']);
            
            jj = 0;
            for j = 1:numel(nuclei_mask)
                
                % Now assign Nuclei instances to this image
                x_start = str2double(nuclei_mask{j}{2}) - str2double(nuclei_trn_img{i}{2}) + 1;
                y_start = str2double(nuclei_mask{j}{3}) - str2double(nuclei_trn_img{i}{3}) + 1;
                x_end = str2double(nuclei_trn_img{i}{2}) + str2double(nuclei_trn_img{i}{4}) - str2double(nuclei_mask{j}{2}) - str2double(nuclei_mask{j}{4});
                y_end = str2double(nuclei_trn_img{i}{3}) + str2double(nuclei_trn_img{i}{5}) - str2double(nuclei_mask{j}{3}) - str2double(nuclei_mask{j}{5});
                
                if (x_start > 0 && y_start > 0 && x_end > 0 && y_end > 0)
                    instances = uint8(zeros(size(img(:,:,1))));
                    mask = imread([imgpath '/' nuclei_mask{j}{7}]);
                    instances(y_start:size(instances,1)-y_end, x_start:size(instances,2)-x_end) = mask(:,:,1);
                    mkdir([nucleus_out_path, name, '/masks_001/']);
                    imwrite(instances, [nucleus_out_path, name, '/masks_001/', name, '_', sprintf('%04d', jj), '.png']);
                    jj = jj+1;
                end
            end
        end
    end
end
    
%% Load tissue training image and create instance labels
if output_list(2)
    for i = 1:numel(tissue_trn_img)
        img = imread([imgpath '/' tissue_trn_img{i}{7}]);
        
        [pathstr, name, ext] = fileparts(tissue_trn_img{i}{7});
         mkdir([tissue_out_path, name, '/images/']);
         imwrite(img, [tissue_out_path, name, '/images/', name, '.png']);

        jj = 0;
        for j = 1:numel(tissue_mask)
            img_prefix = strsplit(tissue_trn_img{i}{7}, '_'); img_prefix = img_prefix{1};
            mask_prefix = strsplit(tissue_mask{j}{7}, '_'); mask_prefix = mask_prefix{1};

            if strcmp(img_prefix, mask_prefix)
                % Now assign Tissue instances to this image
                x_start = str2double(tissue_mask{j}{2}) - str2double(tissue_trn_img{i}{2}) + 1;
                y_start = str2double(tissue_mask{j}{3}) - str2double(tissue_trn_img{i}{3}) + 1;
                x_end = str2double(tissue_trn_img{i}{2}) + str2double(tissue_trn_img{i}{4}) - str2double(tissue_mask{j}{2}) - str2double(tissue_mask{j}{4});
                y_end = str2double(tissue_trn_img{i}{3}) + str2double(tissue_trn_img{i}{5}) - str2double(tissue_mask{j}{3}) - str2double(tissue_mask{j}{5});

                if (x_start > 0 && y_start > 0 && x_end > 0 && y_end > 0)
                    instances = uint8(zeros(size(img(:,:,1))));
                    mask = imread([imgpath '/' tissue_mask{j}{7}]);
                    instances(y_start:size(instances,1)-y_end, x_start:size(instances,2)-x_end) = mask(:,:,1);

                    current_class_id = find(strcmp(class_id, tissue_mask{j}{1}));

                     mkdir([tissue_out_path, name, sprintf('/masks_%03d/', 2)]);
                     mkdir([tissue_out_path, name, sprintf('/masks_%03d/', 3)]);
                     imwrite(instances, [tissue_out_path, name, sprintf('/masks_%03d/', current_class_id), name, sprintf('_%04d', jj), '.png']);
                    jj = jj+1;
                end
            end
        end
    end
end
    
%% Load tissue training image and create instance labels
if output_list(3)
    for i = 1:numel(tissue_trn_img)
        img = imread([imgpath '/' tissue_trn_img{i}{7}]);
        
        [pathstr, name, ext] = fileparts(tissue_trn_img{i}{7});

        jj = 0;
        for j = 1:numel(tissue_mask)
            
            % Now assign Tissue instances to this image
            x_start = str2double(tissue_mask{j}{2}) - str2double(tissue_trn_img{i}{2}) + 1;
            y_start = str2double(tissue_mask{j}{3}) - str2double(tissue_trn_img{i}{3}) + 1;
            str2double(tissue_trn_img{i}{3}) + 1;
            x_end = str2double(tissue_trn_img{i}{2}) + str2double(tissue_trn_img{i}{4}) - str2double(tissue_mask{j}{2}) - str2double(tissue_mask{j}{4});
            y_end = str2double(tissue_trn_img{i}{3}) + str2double(tissue_trn_img{i}{5}) - str2double(tissue_mask{j}{3}) - str2double(tissue_mask{j}{5});
            
            if (x_start > 0 && y_start > 0 && x_end > 0 && y_end > 0)
                instances = uint8(zeros(size(img(:,:,1))));
                mask = imread([imgpath '/' tissue_mask{j}{7}]);
                instances(y_start:size(instances,1)-y_end, x_start:size(instances,2)-x_end) = mask(:,:,1);
                
                current_class_id = find(strcmp(class_id, tissue_mask{j}{1}));
                
                mask3 = repmat(instances,1,1,3);
                avg_col = repmat(mean(mean(img,1),2), size(img,1), size(img,2));
                std_col = repmat(std(std(double(img),[],1),[],2), size(img,1), size(img,2));
                instance_isolated = uint8( uint8(mask3==0).*uint8(avg_col + randn(size(img)).*std_col) + uint8(mask3~=0).*img);
                
                 mkdir([tissue_stage1_out_path, name, sprintf('_%03d', jj), '/images/']);
                 imwrite(instance_isolated, [tissue_stage1_out_path, name, sprintf('_%03d', jj), '/images/', name, sprintf('_%03d', jj), '.png']);
                 mkdir([tissue_stage1_out_path, name, sprintf('_%03d', jj), sprintf('/masks_%03d/', 2)]);
                 mkdir([tissue_stage1_out_path, name, sprintf('_%03d', jj), sprintf('/masks_%03d/', 3)]);
                 imwrite(instances, [tissue_stage1_out_path, name, sprintf('_%03d', jj), sprintf('/masks_%03d/', current_class_id), name, sprintf('_%04d', jj), '.png']);
                jj = jj+1;
            end
        end
    end
end

%% Load tissue training image and create u-net training images
if output_list(4)
    for i = 1:numel(tissue_trn_img)
        
        [pathstr, name, ext] = fileparts(tissue_trn_img{i}{7});
        if ~exist([tissue_unet_out_path, 'train/', name], 'dir') && ~exist([tissue_unet_out_path, 'val/', name], 'dir') % Don't overwrite if folder already in train or val
            
            img = imread([imgpath '/' tissue_trn_img{i}{7}]);

            jj = 1;
            instances = uint8(zeros(size(img(:,:,1))));
            sem_seg = uint8(zeros(size(img(:,:,1))));

            for j = 1:numel(tissue_mask)
                img_prefix = strsplit(tissue_trn_img{i}{7}, '_'); img_prefix = img_prefix{1};
                mask_prefix = strsplit(tissue_mask{j}{7}, '_'); mask_prefix = mask_prefix{1};

                if strcmp(img_prefix, mask_prefix)
                    % Now assign Tissue instances to this image
                    x_start = str2double(tissue_mask{j}{2}) - str2double(tissue_trn_img{i}{2}) + 1;
                    y_start = str2double(tissue_mask{j}{3}) - str2double(tissue_trn_img{i}{3}) + 1;
                    x_end = str2double(tissue_trn_img{i}{2}) + str2double(tissue_trn_img{i}{4}) - str2double(tissue_mask{j}{2}) - str2double(tissue_mask{j}{4});
                    y_end = str2double(tissue_trn_img{i}{3}) + str2double(tissue_trn_img{i}{5}) - str2double(tissue_mask{j}{3}) - str2double(tissue_mask{j}{5});

                    if (x_start > 0 && y_start > 0 && x_end > 0 && y_end > 0)
                        current_class_id = find(strcmp(class_id, tissue_mask{j}{1}));
                        mask = imread([imgpath '/' tissue_mask{j}{7}]);
                        instances(y_start:size(instances,1)-y_end, x_start:size(instances,2)-x_end) = max(mask(:,:,1)/255*jj, instances(y_start:size(instances,1)-y_end, x_start:size(instances,2)-x_end));
                        sem_seg(y_start:size(sem_seg,1)-y_end, x_start:size(sem_seg,2)-x_end) = max(mask(:,:,1)/255*current_class_id, sem_seg(y_start:size(sem_seg,1)-y_end, x_start:size(sem_seg,2)-x_end));

                        jj = jj+1;
                    end
                end
            end

            % Now Generating Boundaries
            boundary = edge(instances,'Roberts',0);
            [xx,yy] = ndgrid(-4:4);
            nhood = sqrt(xx.^2 + yy.^2) <= 3.0;
            boundary = ~imdilate(boundary,nhood);
            sem_seg = uint8(boundary).*sem_seg;
        
            mkdir([tissue_unet_out_path, name]);
            imwrite(img, [tissue_unet_out_path, name, '/', name, '_img.png']);
            imwrite(sem_seg, [tissue_unet_out_path, name, '/', name, '_mask.png']);
        end
    end
end

% figure(1)
% subplot(1,3,1)
% imagesc(img)
% subplot(1,3,2)
% imagesc(instances)
% subplot(1,3,3)
% imagesc(sem_seg)
