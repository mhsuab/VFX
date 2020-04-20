hdr_img = hdrread('result/HDR.hdr');
fprintf('done reading\n');
directory = 'result/';
imwrite(im2uint8(localtonemap(hdr_img, 'RangeCompression', 0.3,'EnhanceContrast', 0.7)), [directory, 'final.jpg'], 'jpg');

