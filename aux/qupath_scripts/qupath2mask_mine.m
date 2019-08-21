clear all
clc
% Function to convert masked images exported from QuPath to instances

% function qupath_to_mask(varargin)
% nucleus_train_val_split = 0.85;
% unet_train_val_split = 0.85;
nuc_output_size = 1024;         % Set this to < 2 to output nuclei images of different sizes for maskrcnn
imgpath = '/Volumes/A-CH-EXDISK/Projects/Project_annotate/masks';
tissue_unet_out_path = '/Volumes/A-CH-EXDISK/Projects/Dataset';
filenames = dir(imgpath); 
filenames = filenames(4:end); %remove hidden system folders
class_id = {'Nucleus', 'Tumor', 'Benign', 'Lumen'};
nuclei_trn_img = {};
nuclei_mask = {};
tissue_trn_img = {};
tissue_mask = {};

downsample = 1.00;
overlap_thresh = 0.0005; %overlap threshold to merge overlapping instances
merge_gland_cls = false; %option to merge tumour and benign class
make_tiles = true; %option to divide images into smaller tiles
tile_side = 1024*uint64(2.0/downsample);
visualize = false;


% check_names = {'EU_18387_14_1E_HandE_TissueTrain_(1.00,66060,20075,6553,4555)',
%                 'EU_18387_14_1E_HandE_TissueTrain_(1.00,76168,22923,5075,3762)',
%                 'EU_26162_16_9x_HandE+-+2017-11-28+11.03.04_TissueTrain_(1.00,71575,40128,7478,9298)',
%                 'EU_30479_16_G_HandE+-+2017-11-28+12.31.57_TissueTrain_(1.00,85805,55583,8204,3965)',
%                 'EU_35928_15_4N_HandE+-+2017-11-28+12.08.09_TissueTrain_(1.00,62504,54130,11631,10039)',
%                 'EU_7189_16_2D_HandE_TissueTrain_(1.00,57611,15937,4797,4449)',
%                 'EU_7189_16_2D_HandE_TissueTrain_(1.00,63607,7966,7080,5364)'};

check_names = {};


%Extract coordinates and images info from tile name
none_file = fopen([tissue_unet_out_path '/' 'Nones.txt'], 'w');
for i = 1:numel(filenames)
    name = filenames(i).name;
    match_str = ['_(\w{3,12})_\(', sprintf('%.2f', downsample), ',(\w{1,6}),(\w{1,6}),(\w{1,6}),(\w{1,6})\)-{0,1}(\w{0,6})'];
    tmp = regexp(name, match_str,'tokens');

    try
        tmp = tmp{1};
    catch
        disp(name)
    end
    tmp{7} = name;
    
    %Populate cell of annotations
    
    if ~strcmp(tmp{6}, 'mask')
        if strcmp(tmp{1}, 'TissueTrain')
            tissue_trn_img{end+1} = tmp;
        end
    else
        if contains(tmp(1), class_id)
            if ~ismember(tmp(1), class_id)
                %Need to modify class id
                for c = 1:numel(class_id)
                    if contains(tmp(1), class_id(c))
                        tmp{1} = class_id{c};
                    end
                end    
            end
            if ~ismember(tmp(1), class_id)
                disp(tmp(1))
                disp("Shouldn't get here")
            end
           tissue_mask{end+1} = tmp;
        elseif strcmp(tmp(1), 'None')
           fprintf(none_file, name)
        elseif ~strcmp(tmp{1}, 'TissueTrain') && ~strcmp(tmp{1}, 'BadPatch')
            disp(tmp(1))
        end
    end
end


    

%% Load tissue training images and create u-net training images
Nimg = numel(tissue_trn_img);
all_img_prefix = num2cell(zeros(1, Nimg));
for i = 1:numel(tissue_trn_img)
    tic
    
    [pathstr, name, ext] = fileparts(tissue_trn_img{i}{7});
    if (~exist([tissue_unet_out_path, '/train/', name], 'dir') && ...
            ~exist([tissue_unet_out_path, '/validate/', name], 'dir') && ...
            ~exist([tissue_unet_out_path, '/test/', name], 'dir')) ...
            || any(strcmp(name, check_names)) % Don't overwrite if folder already in train or val

        img = imread([imgpath '/' tissue_trn_img{i}{7}]);

        jj = 0;
        instances = uint8(zeros(size(img(:,:,1))));
        sem_seg = uint8(zeros(size(img(:,:,1))));
        
        
        img_name_parts = strsplit(tissue_trn_img{i}{7}, '_'); 
        img_prefix = [img_name_parts{1}, '_',  img_name_parts{2}];
        all_img_prefix{i} = img_prefix;
        masks_overlap_with_img = false; %ensure that every images has masks
        for j = 1:numel(tissue_mask) 
            mask_name_parts = strsplit(tissue_mask{j}{7}, '_'); 
            mask_prefix = [mask_name_parts{1}, '_', mask_name_parts{2}];
            
            
            if strcmp(img_prefix, mask_prefix)
                masks_overlap_with_img = true;
                % Now assign Tissue instances to this images
                
                %Load mask
                mask = imread([imgpath '/' tissue_mask{j}{7}]);
                
                %Set boundaries
                x_start = str2double(tissue_mask{j}{2}) - str2double(tissue_trn_img{i}{2}) + 1;
                mask_margins = [1, 0, 1, 0];
                if x_start <= 0               
                    mask_margins(3) = abs(x_start) + 2;
                    x_start = 1;
                end           
                y_start = str2double(tissue_mask{j}{3}) - str2double(tissue_trn_img{i}{3}) + 1;
                if y_start <= 0               
                    mask_margins(1) = abs(y_start) + 2;
                    y_start = 1;
                end  
                x_end = str2double(tissue_trn_img{i}{2}) + str2double(tissue_trn_img{i}{4}) - str2double(tissue_mask{j}{2}) - str2double(tissue_mask{j}{4});
                if x_end <= 0               
                    mask_margins(4) = abs(x_end);
                    x_end = 0;
                end  
                y_end = str2double(tissue_trn_img{i}{3}) + str2double(tissue_trn_img{i}{5}) - str2double(tissue_mask{j}{3}) - str2double(tissue_mask{j}{5});
                if y_end <= 0               
                    mask_margins(2) = abs(y_end);
                    y_end = 0;
                end  
                
                %Redefine mask to match overlap with tile
                mask = mask(mask_margins(1):end-mask_margins(2), mask_margins(3):end-mask_margins(4),1);
                
                %Populate ground truth and instance arrays
                current_class_id = find(strcmp(class_id, tissue_mask{j}{1}));
                if merge_gland_cls && current_class_id == 3
                    current_class_id = 2; %change everything to one gland class 
                end
                
                %Check if any instance is already largely overlapping with
                %images, in which case merge instances
                
                to_write = instances(y_start:size(instances,1)-y_end, x_start:size(instances,2)-x_end);
                if ~isempty(to_write)
                    if sum(mask(mask > 0)) 
                        perc_overlap = sum(to_write(mask > 0) > 0) / sum(mask(mask > 0));
                    else
                        perc_overlap = 0;
                    end
                    if perc_overlap > overlap_thresh
                        iv = mode(to_write(mask > 0 & to_write > 0));
                        instances(y_start:size(instances,1)-y_end, x_start:size(instances,2)-x_end) = max(mask(:,:,1)/255*iv, instances(y_start:size(instances,1)-y_end, x_start:size(instances,2)-x_end));
                    else
                        jj = jj+1; %Increase counter to label new instance
                        instances(y_start:size(instances,1)-y_end, x_start:size(instances,2)-x_end) = max(mask(:,:,1)/255*jj, instances(y_start:size(instances,1)-y_end, x_start:size(instances,2)-x_end));
                    end
                    sem_seg(y_start:size(sem_seg,1)-y_end, x_start:size(sem_seg,2)-x_end) = max(mask(:,:,1)/255*current_class_id, sem_seg(y_start:size(sem_seg,1)-y_end, x_start:size(sem_seg,2)-x_end)); 
                end  
            end
        end
        
        if ~masks_overlap_with_img
            error(str(name) + " has no tissue masks")
        end

        % Now Generating Boundaries
        boundary = edge(instances,'Roberts',0);
        [xx,yy] = ndgrid(-4:4);
        nhood = sqrt(xx.^2 + yy.^2) <= 3.0;
        boundary = ~imdilate(boundary,nhood);
        sem_seg = uint8(boundary).*sem_seg;
        % Change values so that glands are visible
        sem_seg(sem_seg == 2) = 160;
        sem_seg(sem_seg == 3) = 200;
        sem_seg(sem_seg == 4) = 250;
        gt_populate_time = toc;
        display(string(tissue_trn_img{i}{7}) + sprintf(" processed in %.2f seconds", gt_populate_time))
        
        if visualize
            figure
            subplot(1,2,1)
            image(img)
            subplot(1,2,2)
            image(sem_seg)
            pause
        end

        %Sort between train / test / validate splits, 1 path in test and 1 in
        %validate per WSI.
        if sum(strcmp(all_img_prefix, img_prefix)) == 1
           tile_dir = string([tissue_unet_out_path,'/', 'test','/', name]);
        elseif sum(strcmp(all_img_prefix, img_prefix)) == 2
           tile_dir = string([tissue_unet_out_path,'/', 'validate','/', name]);
        else
           tile_dir = string([tissue_unet_out_path,'/', 'train','/', name]); 
        end
        
        %Save img and seg map
        mkdir(char(tile_dir));
        imwrite(img, char(tile_dir + "/" + string(name) + "_img.png"));
        imwrite(sem_seg, char(tile_dir + "/" + string(name) + "_mask.png"));
        
        %Further divide up into tiles that are slightly larger than 1024*1024
        if make_tiles
            
           % SHOULD RESCALE TILES HERE RATHER THAN LATER ?
           subtile_dir = tile_dir + "/tiles";
           mkdir(char(subtile_dir))
           
           %Get full images
           x_tiles = floor(size(img,2) / tile_side);
           y_tiles = floor(size(img,1) / tile_side);
           for x = 1:x_tiles
               for y = 1:y_tiles
                   if y==y_tiles
                    y_idx = (y-1)*tile_side+1:size(img,1); %larger subtile at boundary
                   else
                    y_idx = (y-1)*tile_side+1:y*tile_side;
                   end
                   if x==x_tiles
                    x_idx = (x-1)*tile_side+1:size(img,2); %larger subtile at boundary
                   else
                    x_idx = (x-1)*tile_side+1:x*tile_side;
                   end
                   tile_img = img(y_idx, x_idx, :);
                   tile_gt = sem_seg(y_idx, x_idx, :);
                   imwrite(tile_img, char(subtile_dir + "/" + string(name) + sprintf("_img_%d,%d.png", x_idx(1),y_idx(1))));
                   imwrite(tile_gt, char(subtile_dir + "/" + string(name) + sprintf("_mask_%d,%d.png", x_idx(1),y_idx(1))));
               end 
           end
           
           %+ add random tiles
           clearvars x_idx y_idx
           fill_perc = 0.7; %until 70% of the images is covered again
           fillup = zeros(size(img,1), size(img,2));
           filled = false;
           xlen = size(img,2); ylen = size(img,1);
           while ~filled
               filled = (sum(sum(fillup))/(ylen*xlen) > fill_perc);
               xrs = randi(xlen);  yrs = randi(ylen);
               if yrs + tile_side > ylen
                  yrs = max(ylen - tile_side, 1);
               end
               if xrs + tile_side > xlen
                  xrs = max(xlen - tile_side, 1);
               end
               x_idx = xrs:xrs+min(tile_side, xlen - 1);
               y_idx = yrs:yrs+min(tile_side, ylen - 1);
               tile_img = img(y_idx, x_idx, :);
               tile_gt = sem_seg(y_idx, x_idx, :);
               imwrite(tile_img, char(subtile_dir + "/" + string(name) + sprintf("_img_%d,%d.png", x_idx(1),y_idx(1))));
               imwrite(tile_gt, char(subtile_dir + "/" + string(name) + sprintf("_mask_%d,%d.png", x_idx(1),y_idx(1))));
               fillup(y_idx, x_idx) = true; %update fillup
           end     
           tile_making_time = toc;
           display(sprintf("Tiles saved in %.2f seconds", tile_making_time - gt_populate_time))
        end
    end
end

display("End")
