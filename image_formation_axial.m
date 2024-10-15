clear; clc;
dir = './result_test_image/raw/output/';
dir2 = './result_test_image/raw/output_image/';

row = 512;
col = 128;


for i=1:209
    image_total = zeros(row,col);
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
