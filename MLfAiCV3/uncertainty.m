function [ u ] = uncertainty( p )
p = min(1 - 10e-10, max(10e-10, p));
u = -(p .* log2(p) + (1 - p) .* log2(1 - p));
end

