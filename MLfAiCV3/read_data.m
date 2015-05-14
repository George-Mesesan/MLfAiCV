function [ X, Y ] = read_data( file_prefix )

img_size = 28*28;

fID = fopen(strcat(file_prefix, '-images-idx3-ubyte'), 'r');
X = fread(fID);
X = X(17:end, :);
X = reshape(X, [img_size length(X)/img_size])';
fclose(fID);

fID = fopen(strcat(file_prefix, '-labels-idx1-ubyte'), 'r');
Y = fread(fID);
Y = Y(9:end,:);
fclose(fID);

end

