probe_path = 'D:\study\机器学习\图像生成\baseline_results\CUFS\SCAGAN\Photo';
gallery_path = 'D:\study\机器学习\图像生成\baseline_results\CUFS\GroundTruth\Photo';

probe_list = readImageNames(probe_path);

fsim_value = 0.;
fsim_sum = 0.;
fsim_average = 0.;

for i = 1 : length(probe_list)
    
    probe = imread(fullfile(probe_path, probe_list(i).name));
    
     [height width ch] = size(probe);
    if ch == 3
        probe = rgb2gray(probe);
    end
     probe = double(probe);
    
    gallery =  imread(fullfile(gallery_path, probe_list(i).name));
    
    
     [height width ch] = size(gallery);
    if ch == 3
        gallery = rgb2gray(gallery);
    end
     gallery = double(gallery);
    
    fsim_value = fsim(probe, gallery);
    %fprintf('\n fsim: %f\n', fsim_value)
    fsim_sum = fsim_sum + fsim_value;
end

fsim_average = fsim_sum / length(probe_list);
fprintf('\nAverage fsim is %f\n', fsim_average);
