function y_est = kernel_filter(z, kernel)

k = size(kernel);
k = floor(k(1) / 2);
ndims = length(size(z));
z_ = padarray(z, [0, k, k, 0], 0, 'both');
[N, h, w, c] = size(z_);
y_est = zeros(size(z));
kernel = repmat(kernel, [1, 1, c]);

if ndims == 3
    for n=1:N
        for i=k+1:h
            for j=k+1:w
                window = z_(n, i-k:i+k, j-k:j+k) .* kernel;
                size(window)
                size(y_est(n, i - k, j - k))
                y_est(n, i - k, j - k) = window;
            end
        end
    end
else
  for n=1:N
      for i=k+1:h-k
          for j=k+1:w-k
              window = sum(sum(sum(z_(n, i-k:i+k, j-k:j+k, :) .* kernel)));
              y_est(n, i - k, j - k, :) = window;
          end
      end
  end
end
