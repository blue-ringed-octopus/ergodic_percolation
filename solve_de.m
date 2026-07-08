syms pTp k n c1

a=(n-2)*pTp;
b = 1-pTp;
c = 1;
p = @(k) c1*a^(k-1)+1/((a-b)*(a-1))*(a^(k+1)*(c-b)-b*(c-1)*a^k+a*(b^(k+1)-c)+b*(c-b^k))

eq = p(1) == pTp

c1 = solve(eq, c1 )

p = @(k) simplify(subs(p(k), "c1",c1))

solve(p==2*log(n)/n, k )