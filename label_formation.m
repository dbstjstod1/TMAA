clear; clc;
dir = './result_val/raw/output/';
dir2 = './result_val/raw/output_image/';

row = 96;
col = 65536;
image_total = zeros(row,col*4);

for i=1:1080
    for j=1:4
        fid1=fopen([dir 'output_' num2str(i,'%04d') '_' num2str(j-1,'%d') '.raw'], 'r+');
        I=fread(fid1,row*col,'float32');
        image=reshape(I,row,col);

        image_total(:,1+col*(j-1):col*j) = image;
        fclose(fid1);
    end
    fid2=fopen([dir2 'output_' num2str(i,'%04d') '.raw'],'w');
    cnt=fwrite(fid2,image_total,'float32');
    fclose(fid2);
end
