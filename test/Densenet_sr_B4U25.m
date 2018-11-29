
function Densenet_sr()

setenv('LC_ALL','C')
addpath /home/jishen/my_caffe_gpu/caffe-master;
addpath /home/jishen/my_caffe_gpu/caffe-master/matlab; % change to your caffe path
setenv('GLOG_minloglevel','2')
addpath('/data/jishen/MODN/');
addpath('/data/jishen/MODN/test/evaluation_func/');
addpath('/data/jishen/MODN/test/evaluation_func/matlabPyrTools-master/');
%parameters
gpu_id = 0;
up_scale = 4;
data_set_id =1;
NUM= 82;
interval=6536 ; 

%retrive data
pathfolder = ['/data/jishen/MODN/data/'];
if data_set_id == 1
    % Set5
    setTestCur = 'set5';
    path = [pathfolder setTestCur '/'];
    d = dir(fullfile(path,'*.bmp'));
    filenum=length(d);
    disp(filenum);
end
if data_set_id == 2
    % Set14
    setTestCur = 'Set14';
    path = [pathfolder setTestCur '/'];
     d = dir(fullfile(path,'img_014.png'));
    filenum=length(d);
    disp(filenum)
end
if data_set_id == 3
    % B100
    setTestCur = 'B100';
    path = [pathfolder setTestCur '/'];
    d = dir(fullfile(path,'img_100.png'));
    filenum=length(d);
    disp(filenum);
end
if data_set_id == 4
    % Urban100
    setTestCur = 'Urban100';
    path = [pathfolder setTestCur '/'];
    d = dir(fullfile(path,'img_100_SRF_4_HR.png'));
    filenum=length(d);
    disp(filenum);
end

savepath = ['/data/jishen/MODN/deploy/DensenetB4U25/'];
folderResultCur = fullfile(savepath, [setTestCur,'d_x',num2str(up_scale)]);
if ~exist(folderResultCur,'file')
    mkdir(folderResultCur);
end

caffe.set_mode_gpu(); % for GPU
caffe.set_device(gpu_id);
epochNUM = NUM*interval;
weights = ['/data/jishen/MODN/model/densenet_model_iter_' num2str(epochNUM) '.caffemodel'];
model = '/data/jishen/MODN/test/MODN_B4U25_deploy.prototxt';
im_h_set = cell(filenum,1);
im_gnd_set = cell(filenum,1);
Densenet_set=[];
Bicubic_set=[];
mean_bicubic=[];
mean_densenet=[];
thresh_hei=150;
thresh_wid=150;
rf=16;

for iii = 1:1:length(d)
    disp(['NUM: ' num2str(NUM) '  id: ' num2str(iii)]);
    imageName = d(iii).name;
    imageName = imageName(1:end-4);
    im  = imread([path d(iii).name]);
    
    %% work on luminance only
    im_ycbcr= im;
    %modcrop cb cr channel 
    if size(im,3)>1
        im_ycbcr = rgb2ycbcr(im);
        im_cb = im2double(im_ycbcr(:, :, 2));
        im_cr = im2double(im_ycbcr(:, :, 3));
        im_cb_gnd = modcrop(im_cb, up_scale);
        im_cr_gnd = modcrop(im_cr, up_scale);
        im_cb_b = imresize(imresize(im_cb_gnd,1/up_scale,'bicubic'),up_scale,'bicubic');
        im_cb_b = single(im_cb_b);
        im_cr_b = imresize(imresize(im_cr_gnd,1/up_scale,'bicubic'),up_scale,'bicubic');
        im_cr_b = single(im_cr_b);
    end
    im_y = im2double(im_ycbcr(:, :, 1));
    im_y_gnd = modcrop(im_y, up_scale);
    
    [hei,wid] = size(im_y_gnd);
    im_y_b = imresize(imresize(im_y_gnd,1/up_scale,'bicubic'),up_scale,'bicubic');
    im_y_b = single(im_y_b);
    fprintf('%d\n ',hei,wid);
    %% adaptively spilt
        % decide patch numbers
        hei_patch = ceil(hei/(thresh_hei+rf));
        wid_patch = ceil(wid/(thresh_wid+rf));
        hei_stride = ceil(hei/hei_patch);
        wid_stride = ceil(wid/wid_patch);
        use_start_x=0;
        use_start_y = 0;
        use_end_x = 0;
        use_end_y = 0;
        
        ext_start_x = 0;
        ext_end_x = 0;
        ext_start_y = 0;
        ext_end_y = 0;
        
        posext_start_x = 0;
        posext_start_y = 0;
        posext_end_x = 0;
        posext_end_y = 0;
        
        % extract each patch for inference
        im_y_h = [];
       
        for x = 1 : hei_stride : hei
            for y = 1 : wid_stride : wid
                % decide the length of hei and wid for each patch
                use_start_x = x;
                use_start_y = y;
                if x - rf > 1 % add border
                    ext_start_x = x-rf;
                    posext_start_x = rf+1;
                else
                    ext_start_x = x;
                    posext_start_x = 1;
                end
                if y-rf > 1
                    ext_start_y = y-rf;
                    posext_start_y = rf+1;
                else
                    ext_start_y = y;
                    posext_start_y = 1;
                end
                
                use_end_x = use_start_x+hei_stride-1;
                use_end_y = use_start_y+wid_stride-1;
                
                
                if use_start_x+hei_stride+rf-1 <= hei
                    hei_length = hei_stride+rf;
                    ext_end_x = use_start_x+hei_length-1;
                    posext_end_x = hei_length-rf+posext_start_x-1;
                    
                else
                    hei_length = hei-ext_start_x+1;
                    ext_end_x = ext_start_x+hei_length-1;
                    posext_end_x = hei_length;
                    use_end_x = ext_start_x+hei_length-1;
                end
                if use_start_y+wid_stride+rf-1 <= wid
                    wid_length = wid_stride+rf;
                    ext_end_y = use_start_y+wid_length-1;
                    posext_end_y = wid_length-rf+posext_start_y-1;
                    
                else
                    wid_length = wid-ext_start_y+1;
                    ext_end_y = ext_start_y+wid_length-1;
                    posext_end_y = wid_length;
                    use_end_y = ext_start_y+wid_length-1;
                end
                
                subim_input = im_y_b(ext_start_x : ext_end_x, ext_start_y : ext_end_y);  % input
                data = permute(subim_input,[2, 1, 3]);
                subim_output = do_cnn(model,weights,data);
                subim_output = subim_output';
                subim_output = subim_output(posext_start_x:posext_end_x,posext_start_y:posext_end_y);
                
                % fill im_h with sub_output
                im_y_h(use_start_x:use_end_x,use_start_y:use_end_y) = subim_output;

           
            end
        end
        %% remove border
        im_y_h1 = shave(uint8(single(im_y_h) * 255), [up_scale, up_scale]);
        im_y_gnd1 = shave(uint8(single(im_y_gnd) * 255), [up_scale, up_scale]);
        im_y_b1 = shave(uint8(single(im_y_b) * 255), [up_scale, up_scale]);
        
        if size(im,3) > 1
            im_cb_b1 = shave(uint8(single(im_cb_b) * 255), [up_scale, up_scale]);
            im_cr_b1 = shave(uint8(single(im_cr_b) * 255), [up_scale, up_scale]);
            ycbcr_h = cat(3,(im_y_h1),(im_cb_b1),(im_cr_b1));
            ycbcr_b1= cat(3,(im_y_b1),(im_cb_b1),(im_cr_b1));
            im_h1 = ycbcr2rgb(ycbcr_h);
            im_b1 = ycbcr2rgb(ycbcr_b1);
            
            im_cb_gnd1 = shave(uint8(single(im_cb_gnd) * 255), [up_scale, up_scale]);
            im_cr_gnd1 = shave(uint8(single(im_cr_gnd) * 255), [up_scale, up_scale]);
            ycbcr_gnd = cat(3,(im_y_gnd1),(im_cb_gnd1),(im_cr_gnd1));
            im_gnd1 = ycbcr2rgb(ycbcr_gnd);
        else
            im_h1 = im_y_h1;
            im_gnd1 = im_y_gnd1;
        end
        im_h_set{iii} = im_h1;
        im_gnd_set{iii} = im_gnd1;
        
        imwrite(im_h1,fullfile(folderResultCur,[imageName,'_result','_x',num2str(up_scale),'.png']));
        %imwrite(im_b1,fullfile(folderResultCur,[imageName,'_bicubic','_x',num2str(up_scale),'.png']));
        imwrite(im_gnd1,fullfile(folderResultCur,[imageName,'_groundtruth','_x',num2str(up_scale),'.png']));
        
         %% compute PSNR and SSIM and IFC
        Bic(1) = compute_psnr(im_y_gnd1,im_y_b1);
       Densenet(1) = compute_psnr(im_y_gnd1,im_y_h1);
        Bic(2) = ssim_index(im_y_gnd1,im_y_b1);
       Densenet(2) = ssim_index(im_y_gnd1,im_y_h1);
       Bic(3) = ifcvec(double(im_y_gnd1),double(im_y_b1));
        Densenet(3) = ifcvec(double(im_y_gnd1),double(im_y_h1));
        
        Bicubic_set=[Bicubic_set;Bic];
        Densenet_set=[Densenet_set;Densenet];
        end
    mean_bicubic = [mean_bicubic; [mean(Bicubic_set(:,1)) mean(Bicubic_set(:,2)) mean(Bicubic_set(:,3))]];
    mean_densenet = [mean_densenet; [mean(Densenet_set(:,1)) mean(Densenet_set(:,2)) mean(Densenet_set(:,3))]];

   disp(['epoch: ' num2str(NUM) '---- bic = ' num2str(mean_bicubic(1,:)) '---- densenet = ' num2str(mean_densenet(1,:))]);
   fid=fopen('/data/jishen/MODN/test/B4U25.txt','a');
   formatSpec='%s\n\r';
   fprintf(fid,formatSpec, ['epoch: ' num2str(NUM) '---- bic = ' num2str(mean_bicubic(1,:)) '---- densenet = ' num2str(mean_densenet(1,:))] );
   fclose(fid);
       
   