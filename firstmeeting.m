clear all 
close all
syms s2 positive 

N = 10;
M=sym(zeros(N,N));

for i = 0:N-1
    for j = 0:N-1
        I = j+1;
        n = N-I;
        r = i-j;
        p_meet = 1-(1-s2)^I;
        if i>=j
            M(i+1,j+1) = nchoosek(n,r)*(p_meet)^r*(1-p_meet)^(n-r);
        end
    end 
end 

k=10;
Q = M(1:N-1, 1:N-1);

v= zeros(N-1,1);
v(1)=1;
p = simplify(ones(1,N-1)*Q^k*v)